# -*- coding: utf-8 -*-
"""
glTF 骨骼动画模型加载器 — CPU 蒙皮管线
解析 .glb 中的网格、骨骼层级、蒙皮绑定、动画关键帧。
每帧 advance(dt) → get_skinned_mesh() 输出蒙皮后顶点/法线/索引 供 OpenGL 绘制。

用法:
    model = AnimatedModel("flybird.glb")
    while running:
        model.advance(dt)
        pos, nrm, idx = model.get_skinned_mesh()
        # 用 glBegin(GL_TRIANGLES) 逐面绘制
"""

import numpy as np
import math
from pygltflib import GLTF2

# ════════════════════════════════════════════════════════════
# 矩阵 / 四元数工具
# ════════════════════════════════════════════════════════════

def quat_to_mat(qx, qy, qz, qw):
    """四元数 (x,y,z,w) → 4x4 旋转矩阵 (列主序: v' = M @ v)"""
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, xy, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    return np.array([
        [1.0-(yy+zz),      xy-wz,      xz+wy, 0.0],
        [     xy+wz, 1.0-(xx+zz),      yz-wx, 0.0],
        [     xz-wy,      yz+wx, 1.0-(xx+yy), 0.0],
        [        0.0,         0.0,         0.0, 1.0],
    ], dtype=np.float32)


def trs_to_mat(tx, ty, tz, rx, ry, rz, rw, sx, sy, sz):
    """TRS → 4x4 列主序矩阵: M = T * R * S, v' = M @ v"""
    R = quat_to_mat(rx, ry, rz, rw)
    # 列主序: 缩放作用于 R 的列 (前 3 列)
    R[:, 0] *= sx
    R[:, 1] *= sy
    R[:, 2] *= sz
    # 平移在第四列
    R[0, 3] = tx
    R[1, 3] = ty
    R[2, 3] = tz
    return R


def slerp(q0, q1, t):
    """球面线性插值四元数 (x,y,z,w)"""
    dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3]
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        n = math.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3])
        return r / max(n, 1e-9)
    theta_0 = math.acos(dot)
    sin_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_t = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_t / sin_0
    s1 = sin_t / sin_0
    return s0 * q0 + s1 * q1


# ════════════════════════════════════════════════════════════
# 主类
# ════════════════════════════════════════════════════════════

class AnimatedModel:
    def __init__(self, glb_path: str):
        gltf = GLTF2().load(glb_path)
        data = gltf.binary_blob()

        # ---- 1. 骨骼层级 ----
        nc = len(gltf.nodes)
        self._parent = np.full(nc, -1, dtype=np.int32)
        for i, node in enumerate(gltf.nodes):
            if node.children:
                for c in node.children:
                    self._parent[c] = i
        # 拓扑排序：父节点必须在子节点之前
        self._node_order = self._topo_sort(nc)

        # ---- 2. 节点初始 T/R/S (动画的 fallback 默认值) ----
        self._init_trs = [None] * nc
        for i, node in enumerate(gltf.nodes):
            t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
            r = np.array(node.rotation or [0, 0, 0, 1], dtype=np.float32)
            s = np.array(node.scale or [1, 1, 1], dtype=np.float32)
            self._init_trs[i] = (t, r, s)

        # ---- 3. 找 mesh node + 读网格数据 ----
        self.mesh_node = self._find_mesh_node(gltf)
        self._load_mesh(gltf, data)

        # ---- 4. 蒙皮 ----
        self._load_skin(gltf, data)

        # ---- 5. 动画 ----
        self._load_animations(gltf, data)
        self._active_anim = 0
        self._anim_time = 0.0

        # ---- 6. 运行时缓冲 ----
        self._local_mats = np.tile(np.eye(4, dtype=np.float32), (nc, 1, 1))
        self._global_mats = np.tile(np.eye(4, dtype=np.float32), (nc, 1, 1))
        self._skin_mats = np.zeros((self.joint_count, 4, 4), dtype=np.float32)
        self._sk_pos = np.zeros_like(self._bind_pos, dtype=np.float32)
        self._sk_nrm = np.zeros_like(self._bind_nrm, dtype=np.float32)

        # 模型原始包围盒（绑定姿态），用于归一化缩放
        bmin = self._bind_pos.min(axis=0)
        bmax = self._bind_pos.max(axis=0)
        extent = float(np.max(bmax - bmin))
        self.model_scale = 1.0 / max(extent, 0.01)
        self.model_center = (bmin + bmax) / 2.0

        # 初始化第一帧蒙皮
        self.advance(0.0)

        print(f"[INFO] AnimatedModel loaded: {len(self._bind_pos)}verts {len(self._indices)}tris "
              f"{self.joint_count}joints {len(self._animations)}anims scale={self.model_scale:.2f}")

    # ----------------------------------------------------------
    def _find_mesh_node(self, gltf):
        for i, node in enumerate(gltf.nodes):
            if node.mesh is not None and node.skin is not None:
                return i
        raise ValueError("找不到带 skin 的 mesh node")

    def _topo_sort(self, nc):
        """BFS from roots, guaranteeing parents before children"""
        order = []
        for root in sorted(i for i in range(nc) if self._parent[i] < 0):
            queue = [root]
            while queue:
                curr = queue.pop(0)
                order.append(curr)
                for c in range(nc):
                    if self._parent[c] == curr:
                        queue.append(c)
        return np.array(order, dtype=np.int32)

    # ----------------------------------------------------------
    # 网格数据读取
    # ----------------------------------------------------------
    def _load_mesh(self, gltf, data):
        m = gltf.meshes[gltf.nodes[self.mesh_node].mesh]
        p = m.primitives[0]

        pos = gltf.accessors[p.attributes.POSITION]
        self._bind_pos = self._r(gltf, data, pos).astype(np.float32)

        nrm = gltf.accessors[p.attributes.NORMAL]
        self._bind_nrm = self._r(gltf, data, nrm).astype(np.float32)

        jnt = gltf.accessors[p.attributes.JOINTS_0]
        raw_j = self._r(gltf, data, jnt)
        if raw_j.ndim == 1:
            raw_j = raw_j.reshape(-1, 4)
        self._v_joints = raw_j.astype(np.int32)

        wgt = gltf.accessors[p.attributes.WEIGHTS_0]
        raw_w = self._r(gltf, data, wgt)
        if raw_w.ndim == 1:
            raw_w = raw_w.reshape(-1, 4)
        self._v_weights = raw_w.astype(np.float32)

        idx = gltf.accessors[p.indices]
        raw_i = self._r(gltf, data, idx).astype(np.uint32)
        self._indices = raw_i.reshape(-1, 3)

    # ----------------------------------------------------------
    # 蒙皮
    # ----------------------------------------------------------
    def _load_skin(self, gltf, data):
        node = gltf.nodes[self.mesh_node]
        skin = gltf.skins[node.skin]
        self.joint_count = len(skin.joints)
        self._joint_nodes = np.array(skin.joints, dtype=np.int32)

        ibm = gltf.accessors[skin.inverseBindMatrices]
        raw = self._r(gltf, data, ibm).astype(np.float32).reshape(-1, 4, 4)
        # glTF stores matrices column-major; numpy reshape reads row-major → transpose!
        self._inv_bind = raw.transpose(0, 2, 1)

    # ----------------------------------------------------------
    # 动画
    # ----------------------------------------------------------
    def _load_animations(self, gltf, data):
        self._animations = []
        for anim in gltf.animations:
            chs = []
            for ch in anim.channels:
                s = anim.samplers[ch.sampler]
                times = self._r(gltf, data, gltf.accessors[s.input]).astype(np.float32)
                vals = self._r(gltf, data, gltf.accessors[s.output]).astype(np.float32)
                chs.append({
                    'node': ch.target.node,
                    'path': ch.target.path,
                    'times': times,
                    'values': vals,
                })
            duration = float(chs[0]['times'][-1])
            self._animations.append({'duration': duration, 'channels': chs})

    # ----------------------------------------------------------
    # 二进制读取工具
    # ----------------------------------------------------------
    @staticmethod
    def _r(gltf, data, acc):
        bv = gltf.bufferViews[acc.bufferView]
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        dmap = {5120: 'b', 5121: 'B', 5122: 'h', 5123: 'H', 5125: 'I', 5126: 'f'}
        cc = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[acc.type]
        arr = np.frombuffer(data, dtype=np.dtype(dmap[acc.componentType]),
                           count=acc.count * cc, offset=start)
        if acc.type == 'SCALAR':
            return arr.copy()
        return arr.reshape(-1, cc).copy()

    # ════════════════════════════════════════════════════════
    # 动画控制
    # ════════════════════════════════════════════════════════
    @property
    def animation_count(self): return len(self._animations)

    @property
    def animation_duration(self): return self._animations[self._active_anim]['duration']

    def set_animation(self, i):
        if 0 <= i < len(self._animations):
            self._active_anim = i
            self._anim_time = 0.0

    # ════════════════════════════════════════════════════════
    # 每帧推进
    # ════════════════════════════════════════════════════════
    def advance(self, dt: float):
        """推进动画 dt 秒，CPU 蒙皮到 self._sk_pos / self._sk_nrm"""
        anim = self._animations[self._active_anim]
        dur = anim['duration']
        chs = anim['channels']

        self._anim_time += dt
        if dur > 0:
            self._anim_time %= dur

        # 1) 初始化为 bind TRS
        trs_now = {}
        for ch in chs:
            n = ch['node']
            if n not in trs_now:
                t, r, s = self._init_trs[n]
                trs_now[n] = [t.copy(), r.copy(), s.copy()]

        # 2) 动画插值 (覆盖有动画的节点；未覆盖的保持 init TRS)
        for ch in chs:
            n, path = ch['node'], ch['path']
            times = ch['times']
            vals = ch['values'].ravel()
            ct = self._anim_time

            # 钳位到动画时间范围
            ki = np.searchsorted(times, ct, side='right') - 1
            if ki < 0:
                ki, alpha = 0, 0.0
            elif ki >= len(times) - 1:
                ki, alpha = max(len(times)-2, 0), 1.0
            else:
                ki = max(ki, 0)
                alpha = (ct - times[ki]) / max(times[ki+1] - times[ki], 1e-8)

            if path == 'translation':
                v0 = vals[ki*3:ki*3+3]
                v1 = vals[(ki+1)*3:(ki+1)*3+3]
                trs_now[n][0] = v0 + alpha * (v1 - v0)
            elif path == 'rotation':
                v0 = vals[ki*4:ki*4+4]
                v1 = vals[(ki+1)*4:(ki+1)*4+4]
                trs_now[n][1] = slerp(v0, v1, alpha)
            elif path == 'scale':
                v0 = vals[ki*3:ki*3+3]
                v1 = vals[(ki+1)*3:(ki+1)*3+3]
                trs_now[n][2] = v0 + alpha * (v1 - v0)

        # 3) 汇编局部矩阵
        for i in range(len(self._init_trs)):
            if i in trs_now:
                t, r, s = trs_now[i]
            else:
                t, r, s = self._init_trs[i]
            self._local_mats[i] = trs_to_mat(t[0], t[1], t[2],
                                              r[0], r[1], r[2], r[3],
                                              s[0], s[1], s[2])

        # 4) 骨骼正解 (拓扑序: parent→child)
        for i in self._node_order:
            p = self._parent[i]
            if p < 0:
                self._global_mats[i] = self._local_mats[i].copy()
            else:
                np.matmul(self._global_mats[p], self._local_mats[i], out=self._global_mats[i])

        # 5) 蒙皮矩阵 = global[joint_node] * inverse_bind
        for ji in range(self.joint_count):
            jn = self._joint_nodes[ji]
            np.matmul(self._global_mats[jn], self._inv_bind[ji], out=self._skin_mats[ji])

        # 6) CPU 蒙皮 → 世界空间（armature 空间）
        self._sk_pos.fill(0.0)
        self._sk_nrm.fill(0.0)
        bp, bn = self._bind_pos, self._bind_nrm

        for vi in range(len(bp)):
            p = np.array([bp[vi, 0], bp[vi, 1], bp[vi, 2], 1.0], dtype=np.float32)
            n = np.array([bn[vi, 0], bn[vi, 1], bn[vi, 2], 0.0], dtype=np.float32)
            sp = np.zeros(4, dtype=np.float32)
            sn = np.zeros(4, dtype=np.float32)
            for bi in range(4):
                w = self._v_weights[vi, bi]
                if w < 1e-6:
                    continue
                M = self._skin_mats[self._v_joints[vi, bi]]
                sp += w * (M @ p)
                sn += w * (M @ n)
            self._sk_pos[vi] = sp[:3] / max(sp[3], 1e-9)
            nr = sn[:3]
            nlen = np.linalg.norm(nr)
            if nlen > 1e-9:
                self._sk_nrm[vi] = nr / nlen

    # ════════════════════════════════════════════════════════
    # 获取蒙皮网格
    # ════════════════════════════════════════════════════════
    def get_skinned_mesh(self):
        """返回 (positions_Nx3, normals_Nx3, indices_Mx3)"""
        return self._sk_pos, self._sk_nrm, self._indices
