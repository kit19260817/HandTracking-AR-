# -*- coding: utf-8 -*-
"""
HandTracking AR — 手部追踪 + 3D 模型叠加
摄像头画面作为背景，3D 模型跟随食指指尖移动。

技术栈：MediaPipe(手部追踪) + OpenCV(取流) + pygame/PyOpenGL(AR 渲染)
按 ESC 退出。
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import sys
import os
import traceback
from datetime import datetime

from pygame.locals import (
    QUIT, KEYDOWN, K_ESCAPE, DOUBLEBUF, OPENGL,
)
from OpenGL.GL import *   # noqa: F401,F403
from OpenGL.GLU import *  # noqa: F401,F403
from gltf_animated_model import AnimatedModel

# ============================================================
# 崩溃日志 — 即使 console=False 也能捕获错误
# ============================================================
def _get_log_path():
    """日志写到 exe 所在目录（打包模式）或脚本所在目录（开发模式）"""
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), 'crash_log.txt')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crash_log.txt')

def _crash_handler(exc_type, exc_value, exc_tb):
    """全局异常钩子：把未捕获的异常写到日志文件"""
    with open(_get_log_path(), 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"Frozen: {getattr(sys, 'frozen', False)}\n")
        f.write(f"Python: {sys.version}\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    # 也打印到控制台（如果有控制台的话）
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.exit(1)

sys.excepthook = _crash_handler

# ============================================================
# 工具函数
# ============================================================
def resource_path(*parts):
    """兼容开发模式和 PyInstaller 打包模式的资源路径。
    开发模式：返回脚本所在目录下的相对路径。
    打包模式：返回 _MEIPASS 临时解压目录下的路径。"""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def surface_to_texture(surface):
    """pygame Surface → OpenGL 纹理 ID（y 翻转，纹理坐标 (0,0)=左下 / (1,1)=右上）。
    必须在 OpenGL 上下文创建后调用。"""
    tex_data = pygame.image.tostring(surface, "RGBA", True)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 surface.get_width(), surface.get_height(),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
    return tex_id


def load_font(size, bold=False):
    """安全加载字体，绕过 pygame 2.6.x SysFont 枚举系统字体时的 bug。
    直接用字体文件路径加载微软雅黑，找不到则回退默认字体。"""
    candidates = []
    if bold:
        candidates.append("C:/Windows/Fonts/msyhbd.ttc")   # 雅黑粗体
    candidates.append("C:/Windows/Fonts/msyh.ttc")         # 雅黑常规
    candidates.append("C:/Windows/Fonts/simhei.ttf")       # 黑体兜底
    for path in candidates:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                pass
    return pygame.font.Font(None, size)  # pygame 内置默认（不支持中文但不崩溃）


def hsl_to_rgb(h, s, l):
    """h,s,l ∈ [0,1] → (r,g,b) ∈ [0,1]"""
    if s == 0:
        return (l, l, l)
    def _hue2rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (_hue2rgb(p, q, h + 1/3), _hue2rgb(p, q, h), _hue2rgb(p, q, h - 1/3))


def is_palm_up(hand):
    """检测手掌是否朝上（真实空间中掌心朝天花板）"""
    v1 = (hand[5].x - hand[0].x, hand[5].y - hand[0].y, hand[5].z - hand[0].z)
    v2 = (hand[17].x - hand[0].x, hand[17].y - hand[0].y, hand[17].z - hand[0].z)
    nx = v1[1] * v2[2] - v1[2] * v2[1]
    ny = v1[2] * v2[0] - v1[0] * v2[2]
    nz = v1[0] * v2[1] - v1[1] * v2[0]
    nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nlen > 1e-9:
        ny /= nlen
    return ny > 0.4


def palm_center(hand):
    """返回掌心归一化坐标 (x, y)"""
    ids = [0, 5, 9, 13, 17]
    x = sum(hand[i].x for i in ids) / len(ids)
    y = sum(hand[i].y for i in ids) / len(ids)
    return x, y


def is_index_pointing(hand):
    """检测左手食指指向屏幕手势：食指伸直，中/无名/小指弯曲。
    判断逻辑：每根手指 tip→mcp 距离 / pip→mcp 距离的比值。
    比值 > 1.4 → 伸直，比值 < 1.2 → 弯曲。"""
    def _extended(tip_id, pip_id, mcp_id):
        dx_tm = hand[tip_id].x - hand[mcp_id].x
        dy_tm = hand[tip_id].y - hand[mcp_id].y
        d_tip_mcp = math.sqrt(dx_tm * dx_tm + dy_tm * dy_tm)
        dx_pm = hand[pip_id].x - hand[mcp_id].x
        dy_pm = hand[pip_id].y - hand[mcp_id].y
        d_pip_mcp = math.sqrt(dx_pm * dx_pm + dy_pm * dy_pm)
        if d_pip_mcp < 0.015:
            return False
        return d_tip_mcp / d_pip_mcp > 1.4

    idx_ok = _extended(8, 6, 5)          # 食指伸直
    mid_curled = not _extended(12, 10, 9)  # 中指弯曲
    ring_curled = not _extended(16, 14, 13)  # 无名指弯曲
    pinky_curled = not _extended(20, 18, 17)  # 小指弯曲

    return idx_ok and mid_curled and ring_curled and pinky_curled


def is_thumb_up(hand):
    """左手大拇指朝上（点赞手势），中/无名/小指弯曲"""
    # 拇指朝上：tip(4).y < mcp(2).y（图像中更高 = 向上）
    thumb_up = hand[4].y < hand[2].y - 0.02

    def _curled(tip_id, pip_id, mcp_id):
        dx_tm = hand[tip_id].x - hand[mcp_id].x
        dy_tm = hand[tip_id].y - hand[mcp_id].y
        d_tip = math.sqrt(dx_tm * dx_tm + dy_tm * dy_tm)
        dx_pm = hand[pip_id].x - hand[mcp_id].x
        dy_pm = hand[pip_id].y - hand[mcp_id].y
        d_pip = math.sqrt(dx_pm * dx_pm + dy_pm * dy_pm)
        if d_pip < 0.015:
            return True
        return d_tip / d_pip < 1.2

    return (thumb_up and
            _curled(12, 10, 9) and _curled(16, 14, 13) and _curled(20, 18, 17))


# ============================================================
# 配置区
# ============================================================
WIDTH, HEIGHT = 960, 720          # 窗口 / 背景分辨率
FOV = 60.0                        # 3D 透视视野
CAMERA_Z = 1.0 / math.tan(math.radians(FOV / 2.0))  # 使 z=0 平面纵向覆盖 [-1, 1]

TAU = 0.08                        # 平滑时间常数(秒)，越小越跟手
SPHERE_RADIUS = 0.6              # 模型世界半径（越小越不挡画面）
SPHERE_COLOR = (0.40, 0.75, 1.00) # 淡蓝半透明，AR 全息感
OPACITY = 0.55                    # 模型最大不透明度（0~1，越小越透）

MODEL_PATH = "flybird.glb"         # 带动画的小鸟模型，改为 None 可用回内置小球

# ============================================================
# MediaPipe 手部追踪初始化
# ============================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_FILE = resource_path("models", "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
)
detector = HandLandmarker.create_from_options(options)

# ============================================================
# 动画模型加载
# ============================================================
animated_model = None
if MODEL_PATH:
    model_full_path = resource_path(MODEL_PATH)
    if os.path.exists(model_full_path):
        try:
            animated_model = AnimatedModel(model_full_path)
        except Exception as e:
            print(f"[WARN] 模型加载失败: {e}，将使用内置小球")


def draw_animated_model(model):
    """渲染动画模型当前帧的三角形网格"""
    pos, nrm, idx = model.get_skinned_mesh()
    glBegin(GL_TRIANGLES)
    for tri in idx:
        for vi in tri:
            glNormal3f(nrm[vi, 0], nrm[vi, 1], nrm[vi, 2])
            glVertex3f(pos[vi, 0], pos[vi, 1], pos[vi, 2])
    glEnd()

# ============================================================
# pygame + OpenGL 初始化
# ============================================================
pygame.init()
pygame.display.set_caption("HandTracking AR — ESC 退出")
# 设置窗口图标（替换 pygame 默认黄蛇）
_icon_path = resource_path("models", "icon.png")
if os.path.exists(_icon_path):
    try:
        pygame.display.set_icon(pygame.image.load(_icon_path))
    except Exception as _e:
        print(f"[WARN] icon load failed: {_e}")
screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
glClearColor(0.0, 0.0, 0.0, 1.0)
glEnable(GL_NORMALIZE)
glShadeModel(GL_SMOOTH)

# 背景纹理
bg_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, bg_tex)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# 预分配
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0,
             GL_RGB, GL_UNSIGNED_BYTE, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))

# 模型显示列表
quadric = gluNewQuadric()
gluQuadricNormals(quadric, GLU_SMOOTH)
sphere_list = glGenLists(1)
glNewList(sphere_list, GL_COMPILE)
gluSphere(quadric, SPHERE_RADIUS, 48, 48)
glEndList()

# ============================================================
# UI 手势提示工具栏（顶部居中：图标 + 下方文字标签）
# ============================================================
UI_ICON_SIZE = 56          # 图标边长（像素）
UI_GAP = 64                # 图标间距
UI_PAD = 14                # 工具栏内边距
UI_TOP_MARGIN = 8          # 距窗口顶部

_ui_font = load_font(16, bold=True)

UI_ITEMS = [
    ("palmdown", "显示/隐藏"),
    ("thumbup",  "改变大小"),
    ("pointat",  "改变颜色"),
]

ui_icon_tex = {}     # name -> (tex_id, w, h)
ui_label_tex = {}    # name -> (tex_id, w, h)

for _name, _label in UI_ITEMS:
    _p = resource_path("ui", f"{_name}.png")
    if os.path.exists(_p):
        try:
            _s = pygame.image.load(_p)
            _s = pygame.transform.smoothscale(_s, (UI_ICON_SIZE, UI_ICON_SIZE))
            ui_icon_tex[_name] = (surface_to_texture(_s), UI_ICON_SIZE, UI_ICON_SIZE)
        except Exception as _e:
            print(f"[WARN] UI icon load failed ({_name}): {_e}")
    else:
        print(f"[WARN] UI icon not found: {_p}")
    # 文字标签
    _ls = _ui_font.render(_label, True, (255, 255, 255))
    ui_label_tex[_name] = (surface_to_texture(_ls), _ls.get_width(), _ls.get_height())

# 工具栏几何（OpenGL 正交坐标，y 向上）
_ui_n = len(UI_ITEMS)
_ui_content_w = _ui_n * UI_ICON_SIZE + (_ui_n - 1) * UI_GAP
_ui_bar_w = _ui_content_w + 2 * UI_PAD
_ui_bar_x = (WIDTH - _ui_bar_w) / 2.0
_ui_bar_top = HEIGHT - UI_TOP_MARGIN
_ui_label_h = 22
_ui_bar_h = UI_ICON_SIZE + _ui_label_h + 2 * UI_PAD
_ui_bar_bot = _ui_bar_top - _ui_bar_h
_ui_icon_top = _ui_bar_top - UI_PAD
_ui_icon_bot = _ui_icon_top - UI_ICON_SIZE

# ============================================================
# 摄像头
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("[错误] 摄像头打开失败！请检查：")
    print("  1. 摄像头是否被其他程序占用（Zoom/Teams/微信视频等）")
    print("  2. 摄像头权限是否允许")
    print("  3. 尝试重启电脑后重试")
    # 不退出，让用户看到窗口里的错误提示
else:
    print("[INFO] 摄像头已连接")

# 状态
smooth_x, smooth_y = 0.5, 0.5
visible = 0.0
angle = 0.0
timestamp = 0
clock = pygame.time.Clock()

# 调色状态
model_color = list(SPHERE_COLOR)   # [r, g, b] 实时颜色，左手食指可调
show_colorbar = False              # 是否显示色条
hue_val = 0.56                     # 当前色相 0~1 (初始 = 默认淡蓝)

# 缩放状态
scale_val = 0.5                    # 缩放比例 0~1 (0.5 = 当前默认大小)
show_scalebar = False              # 是否显示缩放条

print("[INFO] AR 窗口已启动，伸出手掌面对摄像头，模型跟随掌心移动。ESC 退出。")

running = True
while running:
    dt = clock.tick(60) / 1000.0
    dt = min(dt, 1.0 / 30.0)

    # ---- 事件 ----
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            running = False

    # ---- 取流 + 追踪 ----
    ret, frame = cap.read()
    if not ret:
        # 摄像头读取失败 — 显示错误提示而不是黑屏
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, WIDTH, HEIGHT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, WIDTH, 0, HEIGHT)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        font = load_font(28)
        err_lines = [
            "[错误] 摄像头读取失败",
            "",
            "可能原因:",
            "  - 摄像头被其他程序占用",
            "  - 摄像头未连接",
            "  - 摄像头权限未授权",
            "",
            "请关闭占用摄像头的程序后重试",
            "或按 ESC 退出",
        ]
        for i, line in enumerate(err_lines):
            surf = font.render(line, True, (255, 100, 100))
            # 用 OpenGL 画文字：先渲染到纹理
            y_pos = HEIGHT - 100 - i * 36
            # pygame 文字转 OpenGL 纹理
            tex_data = pygame.image.tostring(surf, "RGBA", True)
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surf.get_width(), surf.get_height(),
                         0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
            glEnable(GL_TEXTURE_2D)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(WIDTH//2 - surf.get_width()//2, y_pos)
            glTexCoord2f(1, 0); glVertex2f(WIDTH//2 + surf.get_width()//2, y_pos)
            glTexCoord2f(1, 1); glVertex2f(WIDTH//2 + surf.get_width()//2, y_pos + surf.get_height())
            glTexCoord2f(0, 1); glVertex2f(WIDTH//2 - surf.get_width()//2, y_pos + surf.get_height())
            glEnd()
            glDisable(GL_TEXTURE_2D)
            glDeleteTextures([tex])

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        pygame.display.flip()
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if rgb.shape[1] != WIDTH or rgb.shape[0] != HEIGHT:
        rgb = cv2.resize(rgb, (WIDTH, HEIGHT))

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    target_vis = 0.0
    show_colorbar = False
    show_scalebar = False

    if result.hand_landmarks:
        model_hand = None    # 右手 — 手掌朝上 → 控制模型
        color_hand = None    # 左手 — 食指指屏 → 调色
        scale_hand = None    # 左手 — 拇指点赞 → 缩放

        for i, h in enumerate(result.hand_landmarks):
            # MediaPipe handedness: "Left" = 左手, "Right" = 右手
            is_right = (i < len(result.handedness) and
                        result.handedness[i][0].category_name == "Right")
            is_left = (i < len(result.handedness) and
                       result.handedness[i][0].category_name == "Left")

            if is_right and is_palm_up(h):
                model_hand = h
            elif is_left and is_index_pointing(h):
                color_hand = h
            elif is_left and is_thumb_up(h):
                scale_hand = h

        if model_hand is not None:
            nx, ny = palm_center(model_hand)
            target_vis = 1.0
        else:
            nx, ny = smooth_x, smooth_y

        if color_hand is not None:
            show_colorbar = True
            hue_val = color_hand[8].x  # 食指指尖 x 映射为色相 0~1
            cr, cg, cb = hsl_to_rgb(hue_val, 0.85, 0.55)
            model_color = [cr, cg, cb]

        if scale_hand is not None:
            show_scalebar = True
            scale_val = max(0.0, min(1.0, scale_hand[8].x))  # 食指 x 映射 0~1
    else:
        nx, ny = smooth_x, smooth_y

    # ---- 指数平滑（去抖 + 跟手）----
    alpha = 1.0 - math.exp(-dt / TAU)
    smooth_x += (nx - smooth_x) * alpha
    smooth_y += (ny - smooth_y) * alpha
    visible += (target_vis - visible) * (1.0 - math.exp(-dt / 0.12))

    # 归一化坐标 → 世界坐标（z=0 平面）
    aspect = WIDTH / HEIGHT
    wx = (smooth_x - 0.5) * 2.0 * aspect
    wy = (0.5 - smooth_y) * 2.0

    # ---- 渲染 ----
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, WIDTH, HEIGHT)

    # ===== 1) 背景层：摄像头帧纹理 =====
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glDisable(GL_BLEND)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, WIDTH, 0, HEIGHT)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, bg_tex)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)  # 原始纹理，无颜色调制
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT,
                    GL_RGB, GL_UNSIGNED_BYTE, np.ascontiguousarray(rgb))
    glBegin(GL_QUADS)
    # 纹理 y 翻转：OpenGL 纹理原点左下，图像原点左上
    glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 0.0)
    glTexCoord2f(1.0, 1.0); glVertex2f(WIDTH, 0.0)
    glTexCoord2f(1.0, 0.0); glVertex2f(WIDTH, HEIGHT)
    glTexCoord2f(0.0, 0.0); glVertex2f(0.0, HEIGHT)
    glEnd()
    glDisable(GL_TEXTURE_2D)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    # ===== 2) 3D 层：模型跟随食指 =====
    if visible > 0.01:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(FOV, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        gluLookAt(0.0, 0.0, CAMERA_Z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # 光照
        glLightfv(GL_LIGHT0, GL_POSITION, [3.0, 3.0, 4.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.35, 0.38, 0.45, 1.0])

        # 定位 + 轻微漂浮
        float_y = math.sin(pygame.time.get_ticks() * 0.002) * 0.04
        glTranslatef(wx, wy + float_y, 0.0)

        # 静态校正旋转：调整 X/Y/Z 轴和角度让模型立正
        glRotatef(0, 1.0, 0.0, 0.0)   # 绕 X 轴转 90°（把仰卧的模型竖起来）

        # 动态自转：绕 Y 轴匀速旋转
        angle += 55.0 * dt
        glRotatef(angle, 0.0, 1.0, 0.0)

        r, g, b = model_color
        glColor4f(r, g, b, visible * OPACITY)

        if animated_model is not None:
            animated_model.advance(dt)
            # scale_val 0→0.02, 0.5→1.0, 1→2.0
            scale_mult = max(0.02, 2.0 * scale_val)
            s = SPHERE_RADIUS * animated_model.model_scale * scale_mult
            glScalef(s, s, s)
            draw_animated_model(animated_model)
        else:
            glCallList(sphere_list)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

    # ===== 3) 色条叠加 — 另一只手调色时显示 =====
    if show_colorbar:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, WIDTH, 0, HEIGHT)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        BAR_X = 160
        BAR_Y = 50
        BAR_W = 640
        BAR_H = 28
        SEGS = 64

        # 色相渐变条
        glBegin(GL_QUADS)
        for i in range(SEGS):
            h = i / SEGS
            r, g, b = hsl_to_rgb(h, 0.9, 0.55)
            x0 = BAR_X + BAR_W * i / SEGS
            x1 = BAR_X + BAR_W * (i + 1) / SEGS
            glColor3f(r, g, b)
            glVertex2f(x0, BAR_Y)
            glVertex2f(x1, BAR_Y)
            glVertex2f(x1, BAR_Y + BAR_H)
            glVertex2f(x0, BAR_Y + BAR_H)
        glEnd()

        # 白色三角指示器
        ix = BAR_X + hue_val * BAR_W
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_TRIANGLES)
        glVertex2f(ix, BAR_Y + BAR_H + 4)
        glVertex2f(ix - 7, BAR_Y + BAR_H + 16)
        glVertex2f(ix + 7, BAR_Y + BAR_H + 16)
        glEnd()

        glDisable(GL_BLEND)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # ===== 4) 缩放条叠加 — 左手点赞时显示 =====
    if show_scalebar:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, WIDTH, 0, HEIGHT)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        BAR_X = 160
        BAR_Y = 120
        BAR_W = 640
        BAR_H = 28
        SEGS = 64

        # 尺寸渐变条（小→大视觉：灰→白渐变）
        glBegin(GL_QUADS)
        for i in range(SEGS):
            t = i / SEGS  # 0=小(暗) → 1=大(亮)
            glColor3f(0.25 + t * 0.7, 0.55 + t * 0.4, 0.75 + t * 0.25)
            x0 = BAR_X + BAR_W * i / SEGS
            x1 = BAR_X + BAR_W * (i + 1) / SEGS
            glVertex2f(x0, BAR_Y)
            glVertex2f(x1, BAR_Y)
            glVertex2f(x1, BAR_Y + BAR_H)
            glVertex2f(x0, BAR_Y + BAR_H)
        glEnd()

        # 刻度标记（0 / 0.5 / 1.0）
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        for tick in [0.0, 0.5, 1.0]:
            tx = BAR_X + tick * BAR_W
            glVertex2f(tx, BAR_Y - 2)
            glVertex2f(tx, BAR_Y - 8)
        glEnd()

        # 白色三角指示器
        ix = BAR_X + scale_val * BAR_W
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_TRIANGLES)
        glVertex2f(ix, BAR_Y + BAR_H + 4)
        glVertex2f(ix - 7, BAR_Y + BAR_H + 16)
        glVertex2f(ix + 7, BAR_Y + BAR_H + 16)
        glEnd()

        glDisable(GL_BLEND)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # ===== 5) 顶部 UI 工具栏（手势提示图标 + 下方标签）=====
    if ui_icon_tex:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, WIDTH, 0, HEIGHT)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 图标 + 标签
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)

        for i, (_name, _label) in enumerate(UI_ITEMS):
            _cx = _ui_bar_x + UI_PAD + i * (UI_ICON_SIZE + UI_GAP)

            # 图标
            if _name in ui_icon_tex:
                _tex, _iw, _ih = ui_icon_tex[_name]
                glBindTexture(GL_TEXTURE_2D, _tex)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex2f(_cx, _ui_icon_bot)
                glTexCoord2f(1, 0); glVertex2f(_cx + _iw, _ui_icon_bot)
                glTexCoord2f(1, 1); glVertex2f(_cx + _iw, _ui_icon_top)
                glTexCoord2f(0, 1); glVertex2f(_cx, _ui_icon_top)
                glEnd()

            # 文字标签（图标正下方）
            if _name in ui_label_tex:
                _ltex, _lw, _lh = ui_label_tex[_name]
                _lx = _cx + (UI_ICON_SIZE - _lw) / 2.0
                _ly = _ui_icon_bot - _lh - 6
                glBindTexture(GL_TEXTURE_2D, _ltex)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex2f(_lx, _ly)
                glTexCoord2f(1, 0); glVertex2f(_lx + _lw, _ly)
                glTexCoord2f(1, 1); glVertex2f(_lx + _lw, _ly + _lh)
                glTexCoord2f(0, 1); glVertex2f(_lx, _ly + _lh)
                glEnd()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    pygame.display.flip()

cap.release()
pygame.quit()
sys.exit(0)
