@echo off
setlocal EnableDelayedExpansion

call "%~dp0so101_env.bat"
if errorlevel 1 exit /b 1

if "!TELEOP_PORT!"=="" (
    echo [error] TELEOP_PORT is not set. Edit shell\so101_env.bat or set TELEOP_PORT=COM7
    exit /b 1
)

echo [teleoperate] robot=!ROBOT_PORT! teleop=!TELEOP_PORT! camera_key=!CAMERA_KEY! camera_index=!CAMERA_INDEX!

"%PYTHON%" -c "import runpy, sys; import lerobot_camera_opencv_crop; module = sys.argv[1]; sys.argv = [module, *sys.argv[2:]]; runpy.run_module(module, run_name='__main__')" lerobot.scripts.lerobot_teleoperate --robot.type="!ROBOT_TYPE!" --robot.port="!ROBOT_PORT!" --robot.cameras="!ROBOT_CAMERAS!" --teleop.type="!TELEOP_TYPE!" --teleop.port="!TELEOP_PORT!" --display_data=true
exit /b !ERRORLEVEL!
