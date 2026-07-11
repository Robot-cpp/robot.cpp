@echo off
rem Shared SO101 defaults for calibrate / teleoperate / record / client scripts.
rem Edit the values below for your machine (serial ports, camera, server, task, etc.).

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
for %%I in ("%ROOT%\..\..") do set "ROBOT_CPP_ROOT=%%~fI"

rem eval.* needs repo root; camera plugin comes from pip install -e (see environment.yaml).
rem Do not add %ROOT% here — it shadows lerobot_camera_opencv_crop and breaks RealSenseCameraCrop.
set "PYTHONPATH=%ROBOT_CPP_ROOT%;%ROBOT_CPP_ROOT%\robot_client\python;%ROBOT_CPP_ROOT%\robot_client"

rem --- Robot serial ports ---
if not defined ROBOT_PORT set "ROBOT_PORT=?ROBOT_PORT must be set"
if not defined TELEOP_PORT set "TELEOP_PORT=?TELEOP_PORT must be set"
set "ROBOT_TYPE=so101_follower"
set "TELEOP_TYPE=so101_leader"
set "ROBOT_USE_DEGREES=true"

rem --- Camera ---
set "CAMERA_TYPE=realsense"
set CAMERA_RESIZE_WIDTH=224
set CAMERA_RESIZE_HEIGHT=224
set "CAMERA_KEY=camera1"
set "MODEL_IMAGE_NAME=observation.images.camera1"

if /I "%CAMERA_TYPE%"=="iphone" (
    set "CAMERA_DRIVER=opencv_crop"
    set CAMERA_INDEX=0
    set CAMERA_WIDTH=1280
    set CAMERA_HEIGHT=720
    set "CAMERA_BACKEND=AVFOUNDATION"
    set CAMERA_FPS=30
    set CAMERA_WARMUP_S=5
    goto :camera_done
)
if /I "%CAMERA_TYPE%"=="realsense" (
    set "CAMERA_DRIVER=realsense"
    if not defined REALSENSE_SERIAL set "REALSENSE_SERIAL=?REALSENSE_SERIAL must be set"
    set CAMERA_WIDTH=640
    set CAMERA_HEIGHT=480
    set CAMERA_FPS=30
    set "CAMERA_BACKEND=DSHOW"
    set CAMERA_WARMUP_S=5
    goto :camera_done
)
echo [error] unsupported CAMERA_TYPE=%CAMERA_TYPE% (use iphone or realsense)
exit /b 1

:camera_done
if not defined OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS set "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0"

if not defined ROBOT_CAMERAS goto :build_robot_cameras
python -c "import json, os; json.loads(os.environ['ROBOT_CAMERAS'])" 2>nul
if errorlevel 1 goto :build_robot_cameras
goto :robot_cameras_ready

:build_robot_cameras
set "ROBOT_CAMERAS_JSON=%TEMP%\robot_cpp_robot_cameras_%RANDOM%.json"
python "%ROOT%\script\shell\build_robot_cameras.py" > "%ROBOT_CAMERAS_JSON%"
if errorlevel 1 exit /b 1
set /p ROBOT_CAMERAS=<"%ROBOT_CAMERAS_JSON%"
del /f /q "%ROBOT_CAMERAS_JSON%" 2>nul
if not defined ROBOT_CAMERAS exit /b 1

:robot_cameras_ready

rem --- Inference client ---
if not defined ROBOT_PLATFORM set "ROBOT_PLATFORM=lerobot_so101"
if not defined SERVER set "SERVER=127.0.0.1:5555"
if not defined TASK set "TASK=grab the block."
set FPS=25

exit /b 0
