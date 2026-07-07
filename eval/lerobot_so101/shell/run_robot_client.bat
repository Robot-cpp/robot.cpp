@echo off
setlocal EnableDelayedExpansion

call "%~dp0so101_env.bat"
if errorlevel 1 exit /b 1

echo [robot_sync] platform=!ROBOT_PLATFORM! server=!SERVER!
echo [robot_sync] robot_port=!ROBOT_PORT! teleop_port=!TELEOP_PORT! camera_key=!CAMERA_KEY! camera_index=!CAMERA_INDEX! fps=!FPS!
echo [robot_sync] Start model-server first, e.g.:
echo   robot_server\shell\launch_robot_server_windows_cuda.bat

"%PYTHON%" "%ROBOT_CPP_ROOT%\eval\lerobot_so101\run_sync.py"
exit /b !ERRORLEVEL!
