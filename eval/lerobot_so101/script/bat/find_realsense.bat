@echo off
setlocal EnableDelayedExpansion

call "%~dp0so101_env.bat"
if errorlevel 1 exit /b 1

echo == Intel RealSense devices (pyrealsense2) ==
python -m lerobot.scripts.lerobot_find_cameras realsense
echo.
echo Set REALSENSE_SERIAL in script\bat\so101_env.bat, then re-run camera test.
exit /b 0
