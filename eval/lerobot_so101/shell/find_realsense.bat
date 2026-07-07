@echo off
setlocal EnableDelayedExpansion

call "%~dp0so101_env.bat"
if errorlevel 1 exit /b 1

echo == Intel RealSense devices (pyrealsense2) ==
"%PYTHON%" -m lerobot.scripts.lerobot_find_cameras realsense
echo.
echo Set REALSENSE_SERIAL to the serial number above, then re-run camera test.
exit /b 0
