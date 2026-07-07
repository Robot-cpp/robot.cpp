@echo off

rem Shared SO101 defaults for Windows (calibrate / teleoperate / record / client).
rem RealSense: OpenCV DSHOW shows SMPTE color bars only; use pyrealsense2 (realsense_crop).

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
for %%I in ("%ROOT%\..\..") do set "ROBOT_CPP_ROOT=%%~fI"

if not defined CONDA_ENV set "CONDA_ENV=lerobot-demo"
if not defined PYTHON (
    if exist "D:\anaconda3\envs\%CONDA_ENV%\python.exe" (
        set "PYTHON=D:\anaconda3\envs\%CONDA_ENV%\python.exe"
    ) else if exist "%USERPROFILE%\anaconda3\envs\%CONDA_ENV%\python.exe" (
        set "PYTHON=%USERPROFILE%\anaconda3\envs\%CONDA_ENV%\python.exe"
    ) else if exist "%USERPROFILE%\miniconda3\envs\%CONDA_ENV%\python.exe" (
        set "PYTHON=%USERPROFILE%\miniconda3\envs\%CONDA_ENV%\python.exe"
    ) else (
        set "PYTHON=python"
    )
)

set "PYTHONPATH=%ROBOT_CPP_ROOT%;%ROOT%;%ROOT%\lerobot_camera_opencv_crop;%ROBOT_CPP_ROOT%\robot_client\python;%ROBOT_CPP_ROOT%\robot_client"

if not defined ROBOT_PORT set "ROBOT_PORT=COM10"
if not defined TELEOP_PORT set "TELEOP_PORT=COM7"
set "ROBOT_TYPE=so101_follower"
set "TELEOP_TYPE=so101_leader"
set "ROBOT_USE_DEGREES=true"

if not defined CAMERA_KEY set "CAMERA_KEY=camera1"
if not defined MODEL_IMAGE_NAME set "MODEL_IMAGE_NAME=observation.images.camera1"
if not defined CAMERA_DRIVER set "CAMERA_DRIVER=realsense"
if not defined REALSENSE_SERIAL set "REALSENSE_SERIAL=141722072266"
if not defined CAMERA_INDEX set "CAMERA_INDEX=3"
if not defined CAMERA_WIDTH set "CAMERA_WIDTH=640"
if not defined CAMERA_HEIGHT set "CAMERA_HEIGHT=480"
if not defined CAMERA_FPS set "CAMERA_FPS=30"
if not defined CAMERA_BACKEND set "CAMERA_BACKEND=DSHOW"
if not defined CAMERA_RESIZE_WIDTH set "CAMERA_RESIZE_WIDTH=224"
if not defined CAMERA_RESIZE_HEIGHT set "CAMERA_RESIZE_HEIGHT=224"
if not defined CAMERA_WARMUP_S set "CAMERA_WARMUP_S=5"
if not defined OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS set "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0"

if not defined ROBOT_CAMERAS goto :build_robot_cameras
"%PYTHON%" -c "import json, os; json.loads(os.environ['ROBOT_CAMERAS'])" 2>nul
if errorlevel 1 goto :build_robot_cameras
goto :robot_cameras_ready

:build_robot_cameras
set "ROBOT_CAMERAS_JSON=%TEMP%\robot_cpp_robot_cameras_%RANDOM%.json"
"%PYTHON%" "%ROOT%\shell\build_robot_cameras.py" > "%ROBOT_CAMERAS_JSON%"
if errorlevel 1 exit /b 1
set /p ROBOT_CAMERAS=<"%ROBOT_CAMERAS_JSON%"
del /f /q "%ROBOT_CAMERAS_JSON%" 2>nul
if not defined ROBOT_CAMERAS exit /b 1

:robot_cameras_ready

if not defined ROBOT_PLATFORM set "ROBOT_PLATFORM=lerobot_so101"
if not defined SERVER set "SERVER=127.0.0.1:5555"
if not defined TASK set "TASK=grab the block."
if not defined FPS set "FPS=25"

exit /b 0
