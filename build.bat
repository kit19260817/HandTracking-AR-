@echo off
cd /d "%~dp0"

echo ==================================================
echo    HandTracking AR - Build Tool
echo ==================================================
echo.
echo Mode:
echo   [1] Debug  (with console window)
echo   [2] Release (no console window)
echo   [3] Exit
echo.
set /p choice="Select (1/2/3): "

if "%choice%"=="3" exit /b 0
if "%choice%"=="1" (
    set "SPEC_MODE=debug"
) else if "%choice%"=="2" (
    set "SPEC_MODE=release"
) else (
    echo Invalid choice
    pause
    exit /b 1
)

echo.
echo [1/4] Cleaning old files...
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist crash_log.txt del /f crash_log.txt 2>nul
echo Done.
echo.

echo [2/4] Checking PyInstaller...
venv\Scripts\python.exe -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    venv\Scripts\pip.exe install pyinstaller
    if errorlevel 1 (
        echo [ERROR] PyInstaller install failed!
        pause
        exit /b 1
    )
)
echo Done.
echo.

echo [3/4] Configuring spec (%SPEC_MODE%)...
if "%SPEC_MODE%"=="debug" (
    venv\Scripts\python.exe -c "import re; s=open('main.spec','r',encoding='utf-8').read(); s=re.sub(r'console\s*=\s*\w+', 'console=True', s); open('main.spec','w',encoding='utf-8').write(s)"
    echo console=True
) else (
    venv\Scripts\python.exe -c "import re; s=open('main.spec','r',encoding='utf-8').read(); s=re.sub(r'console\s*=\s*\w+', 'console=False', s); open('main.spec','w',encoding='utf-8').write(s)"
    echo console=False
)
echo.

echo [4/4] Building... This may take 3-10 minutes.
echo Collecting MediaPipe / OpenCV / OpenGL deps...
echo Please wait, do not close this window.
echo.
venv\Scripts\python.exe -m PyInstaller main.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ============================================
    echo [ERROR] Build failed! Check error above.
    echo ============================================
    pause
    exit /b 1
)
echo.
echo ============================================
echo  Build complete!
echo ============================================
echo.
echo  Output: dist\main.exe
echo.

for %%A in (dist\main.exe) do echo  Size: %%~zA bytes
echo.

if "%SPEC_MODE%"=="debug" (
    echo [Debug] If exe shows black screen or crashes:
    echo   - Check crash_log.txt next to the exe
    echo   - Console window will show error messages
) else (
    echo [Release] You can send dist\main.exe to others.
    echo   If issues, rebuild with Debug mode to diagnose.
)
echo.
pause
