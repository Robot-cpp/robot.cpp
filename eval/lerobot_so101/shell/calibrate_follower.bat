@echo off
setlocal EnableDelayedExpansion

call "%~dp0so101_env.bat"
if errorlevel 1 exit /b 1

echo [calibrate] follower port=!ROBOT_PORT!

"%PYTHON%" -c "import runpy, sys; import lerobot_camera_opencv_crop; module = sys.argv[1]; sys.argv = [module, *sys.argv[2:]]; runpy.run_module(module, run_name='__main__')" lerobot.scripts.lerobot_calibrate --robot.type="!ROBOT_TYPE!" --robot.port="!ROBOT_PORT!"
exit /b !ERRORLEVEL!
