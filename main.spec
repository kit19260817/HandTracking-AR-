# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files

datas = []
binaries = []
hiddenimports = []

# ---- MediaPipe（最关键，依赖大量二进制文件）----
tmp = collect_all('mediapipe')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# ---- OpenCV ----
tmp = collect_all('cv2')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# ---- OpenGL ----
tmp = collect_all('OpenGL')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# ---- numpy 动态库 ----
binaries += collect_dynamic_libs('numpy')

# ---- pygame 数据文件 ----
datas += collect_data_files('pygame')

# ---- pygltflib（glTF 模型解析库）----
tmp = collect_all('pygltflib')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# ---- 手动添加项目资源文件 ----
datas += [
    ('flybird.glb', '.'),
    ('models/hand_landmarker.task', 'models'),
    ('models/icon.png', 'models'),
    ('ui/palmdown.png', 'ui'),
    ('ui/thumbup.png', 'ui'),
    ('ui/pointat.png', 'ui'),
]

# ---- 确保本地模块被包含 ----
hiddenimports += ['gltf_animated_model']

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,           # 先开控制台调试，确认没问题后改 False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
