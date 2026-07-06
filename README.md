# HandTracking AR 手部追踪增强现实

A desktop AR demo where 3D animated models follow your hand in real time through a webcam.<br>
Uses MediaPipe hand tracking with PyOpenGL rendering, supporting dual-hand gestures for model display, color adjustment, and scaling.<br>
Packaged as a standalone .exe — no Python installation required.<br>
>  
一款基于摄像头的桌面 AR 演示，3D 动画模型实时跟随手掌移动。<br>
使用 MediaPipe 手部追踪 + PyOpenGL 渲染，支持双手手势控制模型显示、调色、缩放。<br>
已打包为独立 exe，无需安装 Python 环境即可运行。

# WIP 仍在开发中
This project is still being actively tested and optimized to improve the user experience.<br>
>  
该项目仍在持续测试并优化用户体验中。

# Usage Notes 使用注意事项
- Download `main.exe` from the Releases section and double-click to run
- A webcam is required; close other apps that may occupy the camera (Zoom, Teams, WeChat video, etc.)
- Press `ESC` to exit
- Right palm facing up → model appears and follows your palm
- Left index finger pointing at screen → color adjustment bar appears
- Left thumb up (thumbs-up gesture) → scale adjustment bar appears
>  
- 在 Release 中下载 `main.exe`，双击即可运行
- 需要摄像头；请关闭可能占用摄像头的程序（Zoom、Teams、微信视频等）
- 按 `ESC` 退出
- 右手掌心朝上 → 模型出现并跟随掌心移动
- 左手食指指向屏幕 → 弹出颜色调节条
- 左手大拇指朝上（点赞手势）→ 弹出大小调节条

# Functions 功能
- Real-time hand tracking with 21 landmarks via MediaPipe
- 3D animated glTF model rendering with CPU-based skeletal skinning
- Dual-hand gesture control: right hand for model display, left hand for color/scale
- HSL color bar for real-time model color adjustment
- Scale bar for model size adjustment (0–1 range)
- Webcam feed as AR background with orthogonal projection
- One-click PyInstaller packaging into standalone .exe
>  
- 基于 MediaPipe 的 21 点实时手部追踪
- glTF 动画模型渲染，CPU 骨骼蒙皮
- 双手手势控制：右手控制模型显示，左手调色 / 缩放
- HSL 色条实时调节模型颜色
- 缩放条调节模型大小（0–1 范围）
- 摄像头画面作为 AR 背景（正交投影）
- PyInstaller 一键打包为独立 exe

# Framework 框架
- MediaPipe: Google's hand landmark detection model (21 keypoints, video mode)
- OpenCV: Webcam capture and frame preprocessing
- PyOpenGL: 3D perspective rendering + 2D orthogonal HUD overlay
- pygame: Window management, input handling, font rendering
- pygltflib: glTF 2.0 binary parsing and skeletal animation
- PyInstaller: Single-file .exe packaging with embedded resources
>  
- MediaPipe: Google 手部关键点检测模型（21 点，视频模式）
- OpenCV: 摄像头取流与帧预处理
- PyOpenGL: 3D 透视渲染 + 2D 正交 HUD 叠加
- pygame: 窗口管理 / 输入处理 / 字体渲染
- pygltflib: glTF 2.0 二进制解析与骨骼动画
- PyInstaller: 单文件 exe 打包，资源内嵌

# Features 特点
- CPU-based skeletal skinning (no GPU shader dependency) — runs on any machine
- Quaternion slerp interpolation for smooth animation blending
- Topological sort for bone hierarchy transform order
- Dual projection pipeline: orthogonal for camera background + perspective for 3D overlay
- GL_REPLACE texture environment to preserve original frame colors (avoids green tint from GL_MODULATE)
- resource_path() helper for seamless dev / PyInstaller mode resource loading
- Exponential smoothing for jitter-free hand tracking (configurable tau)
>  
- CPU 骨骼蒙皮（不依赖 GPU shader），任意机器可运行
- 四元数 slerp 插值实现平滑动画过渡
- 骨骼层级拓扑排序确保变换顺序正确
- 双投影管线：正交投影渲染摄像头背景 + 透视投影叠加 3D 模型
- GL_REPLACE 纹理环境保持原始画面颜色（避免 GL_MODULATE 导致的偏绿）
- resource_path() 辅助函数，兼容开发模式与 PyInstaller 打包模式
- 指数平滑去抖，手部追踪稳定流畅（tau 可调）
