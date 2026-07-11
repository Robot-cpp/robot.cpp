@echo off
setlocal EnableDelayedExpansion

for %%I in ("%~dp0..") do set "TEST_ROOT=%%~fI"
call "%TEST_ROOT%\script\bat\so101_env.bat"
if errorlevel 1 exit /b 1

if not defined FRAMES set "FRAMES=30"
rem lerobot-demo ships opencv-python-headless (no GUI). Default to no preview on Windows.
if not defined PREVIEW set "PREVIEW=0"

if /I "%~1"=="--list-cameras" (
    python "%TEST_ROOT%\test\test_camera.py" --list-cameras %*
    exit /b !ERRORLEVEL!
)

if /I "%~1"=="--probe" (
    python "%TEST_ROOT%\test\test_camera.py" --probe --camera-key "!CAMERA_KEY!" --robot-cameras "!ROBOT_CAMERAS!" %*
    exit /b !ERRORLEVEL!
)

echo [camera-test] camera_key=!CAMERA_KEY! index=!CAMERA_INDEX! frames=!FRAMES!

if "!PREVIEW!"=="0" (
    set "PREVIEW_FLAG=--no-preview"
) else (
    set "PREVIEW_FLAG=--preview"
)

python "%TEST_ROOT%\test\test_camera.py" --camera-key "!CAMERA_KEY!" --camera-index "!CAMERA_INDEX!" --frames !FRAMES! !PREVIEW_FLAG! --robot-cameras "!ROBOT_CAMERAS!" %*
exit /b !ERRORLEVEL!
