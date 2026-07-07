@echo off
setlocal EnableDelayedExpansion

call "%~dp0so101_env.bat"
if errorlevel 1 exit /b 1

if "!TELEOP_PORT!"=="" (
    echo [error] TELEOP_PORT is not set. Edit shell\so101_env.bat or set TELEOP_PORT=COM7
    exit /b 1
)

if not defined DATASET_REPO_ID set "DATASET_REPO_ID=local/so101_record"
if not defined DATASET_ROOT set "DATASET_ROOT=!ROOT!\datasets\so101"
if not defined DATASET_PUSH_TO_HUB set "DATASET_PUSH_TO_HUB=false"
if not defined NUM_EPISODES set "NUM_EPISODES=10"
if not defined SINGLE_TASK set "SINGLE_TASK=grab the block."
if not defined DATASET_FPS set "DATASET_FPS=25"
if not defined EPISODE_TIME_S set "EPISODE_TIME_S=60"
if not defined RESET_TIME_S set "RESET_TIME_S=10"
if not defined STREAMING_ENCODING set "STREAMING_ENCODING=false"
if not defined ENCODER_THREADS set "ENCODER_THREADS=4"
if not defined DISPLAY_DATA set "DISPLAY_DATA=true"

if not exist "!DATASET_ROOT!" mkdir "!DATASET_ROOT!"

echo [record] repo_id=!DATASET_REPO_ID! root=!DATASET_ROOT! episodes=!NUM_EPISODES!

"%PYTHON%" -c "import runpy, sys; import lerobot_camera_opencv_crop; module = sys.argv[1]; sys.argv = [module, *sys.argv[2:]]; runpy.run_module(module, run_name='__main__')" lerobot.scripts.lerobot_record --robot.type="!ROBOT_TYPE!" --robot.port="!ROBOT_PORT!" --robot.cameras="!ROBOT_CAMERAS!" --teleop.type="!TELEOP_TYPE!" --teleop.port="!TELEOP_PORT!" --dataset.repo_id="!DATASET_REPO_ID!" --dataset.root="!DATASET_ROOT!" --dataset.push_to_hub="!DATASET_PUSH_TO_HUB!" --dataset.num_episodes="!NUM_EPISODES!" --dataset.single_task="!SINGLE_TASK!" --dataset.fps="!DATASET_FPS!" --dataset.episode_time_s="!EPISODE_TIME_S!" --dataset.reset_time_s="!RESET_TIME_S!" --dataset.streaming_encoding="!STREAMING_ENCODING!" --dataset.encoder_threads="!ENCODER_THREADS!" --display_data="!DISPLAY_DATA!"
exit /b !ERRORLEVEL!
