@echo off
setlocal EnableDelayedExpansion

call "%~dp0so101_env.bat"
if errorlevel 1 exit /b 1

if "!TELEOP_PORT!"=="" (
    echo [error] TELEOP_PORT is not set. Edit script\bat\so101_env.bat or set TELEOP_PORT=COM7
    exit /b 1
)

echo [calibrate] leader port=!TELEOP_PORT!

python -c "import runpy, sys; import lerobot_camera_opencv_crop; module = sys.argv[1]; sys.argv = [module, *sys.argv[2:]]; runpy.run_module(module, run_name='__main__')" lerobot.scripts.lerobot_calibrate --teleop.type="!TELEOP_TYPE!" --teleop.port="!TELEOP_PORT!"
exit /b !ERRORLEVEL!
