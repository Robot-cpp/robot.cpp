@echo off
setlocal enabledelayedexpansion

rem Launch model-server and run benchmark_latency.py on Windows.
rem Mirrors robot_server/test/test_server_latency.sh.
rem
rem Usage:
rem   set ROBOT_CPP_ROOT=C:\path\to\robot.cpp
rem   set GGUF_DIR=C:\path\to\robot.cpp\ckpts\...
rem   robot_server\test\test_server_latency_windows.bat windows-cuda pi0
rem
rem Thread sweep (4, 8, 12, 16 by default):
rem   set THREADS_SWEEP=1
rem   robot_server\test\test_server_latency_windows.bat windows-cuda pi0

set "SCRIPT_DIR=%~dp0"
if not defined ROBOT_CPP_ROOT (
    set "ROBOT_CPP_ROOT=!SCRIPT_DIR!..\.."
    for %%I in ("!ROBOT_CPP_ROOT!") do set "ROBOT_CPP_ROOT=%%~fI"
)

if not defined GGUF_DIR (
    echo error: GGUF_DIR must be set >&2
    exit /b 1
)

rem ====== change these if needed ======
if not defined BACKEND set "BACKEND=windows-cuda"
set "ARG1=%~1"
set "ARG2=%~2"
if /I "!ARG1!"=="windows-cuda" (
    set "BACKEND=windows-cuda"
    set "ARG1=!ARG2!"
) else if /I "!ARG1!"=="windows-cpu" (
    set "BACKEND=windows-cpu"
    set "ARG1=!ARG2!"
)

if not "!ARG1!"=="" (
    set "MODEL_TYPE=!ARG1!"
) else if not defined MODEL_TYPE (
    set "MODEL_TYPE=smolvla"
)

if /I "!BACKEND!"=="windows-cuda" (
    if not defined BUILD_DIR set "BUILD_DIR=!ROBOT_CPP_ROOT!\build-win-cuda"
    if not defined ARTIFACT_DIR set "ARTIFACT_DIR=!ROBOT_CPP_ROOT!\debug\artifacts\robot_server_latency_windows_cuda"
    set "LAUNCH_BAT=!ROBOT_CPP_ROOT!\robot_server\shell\launch_robot_server_windows_cuda.bat"
) else if /I "!BACKEND!"=="windows-cpu" (
    if not defined BUILD_DIR set "BUILD_DIR=!ROBOT_CPP_ROOT!\build-win-cpu-mingw"
    if not defined ARTIFACT_DIR set "ARTIFACT_DIR=!ROBOT_CPP_ROOT!\debug\artifacts\robot_server_latency_windows_cpu"
    set "LAUNCH_BAT=!ROBOT_CPP_ROOT!\robot_server\shell\launch_robot_server_windows_cpu.bat"
) else (
    echo error: unsupported BACKEND=!BACKEND! >&2
    exit /b 1
)

if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=5569"
if not defined THREADS set "THREADS=8"
if not defined THREADS_SWEEP set "THREADS_SWEEP=0"
if not defined THREADS_MIN set "THREADS_MIN=4"
if not defined THREADS_MAX set "THREADS_MAX=16"
if not defined THREADS_STEP set "THREADS_STEP=4"
if not defined PROMPT set "PROMPT=grab the block."
if /I "!MODEL_TYPE!"=="smolvla" (
    set "DEFAULT_IMAGE_NAMES=observation.images.front"
    set "DEFAULT_STATE_DIM=6"
) else if /I "!MODEL_TYPE!"=="pi0" (
    set "DEFAULT_IMAGE_NAMES=observation.images.image,observation.images.image2"
    set "DEFAULT_STATE_DIM=32"
) else (
    set "DEFAULT_IMAGE_NAMES=image"
    set "DEFAULT_STATE_DIM=6"
)
if not defined IMAGE_NAMES (
    if defined IMAGE_NAME (
        set "IMAGE_NAMES=!IMAGE_NAME!"
    ) else (
        set "IMAGE_NAMES=!DEFAULT_IMAGE_NAMES!"
    )
)
if not defined IMAGE_WIDTH set "IMAGE_WIDTH=224"
if not defined IMAGE_HEIGHT set "IMAGE_HEIGHT=224"
if not defined STATE_DIM set "STATE_DIM=!DEFAULT_STATE_DIM!"
if not defined WARMUP set "WARMUP=5"
if not defined LOOPS set "LOOPS=100"
if not defined SERVER_WAIT_S set "SERVER_WAIT_S=120"
if not defined DTYPE set "DTYPE=f32"

if not defined PYTHON set "PYTHON=python"
rem ====================================

set "BENCHMARK_SCRIPT=!ROBOT_CPP_ROOT!\robot_server\test\benchmark_latency.py"
set "RESULT_TSV=!ARTIFACT_DIR!\benchmark_server.tsv"

if not exist "!LAUNCH_BAT!" (
    echo error: launch script not found: !LAUNCH_BAT! >&2
    exit /b 1
)
if not exist "!BENCHMARK_SCRIPT!" (
    echo error: benchmark script not found: !BENCHMARK_SCRIPT! >&2
    exit /b 1
)

set "BENCHMARK_IMAGE_ARGS="
set "IMAGE_NAME_LIST=!IMAGE_NAMES:,= !"
for %%I in (!IMAGE_NAME_LIST!) do (
    if not "%%I"=="" set "BENCHMARK_IMAGE_ARGS=!BENCHMARK_IMAGE_ARGS! --image-name %%I"
)

echo == prepare outputs ==
if not exist "!ARTIFACT_DIR!" mkdir "!ARTIFACT_DIR!"
if exist "!RESULT_TSV!" del /f /q "!RESULT_TSV!"

if "!THREADS_SWEEP!"=="1" (
    echo == thread sweep: !THREADS_MIN!..!THREADS_MAX! step !THREADS_STEP! ==
    set "T=!THREADS_MIN!"
    goto :sweep_loop
)

call :run_benchmark !THREADS!
if errorlevel 1 exit /b !ERRORLEVEL!
goto :finish

:sweep_loop
if !T! GTR !THREADS_MAX! goto :finish
call :run_benchmark !T!
if errorlevel 1 exit /b !ERRORLEVEL!
set /A T+=THREADS_STEP
goto :sweep_loop

:finish
echo == done ==
echo backend: !BACKEND!
echo result tsv: !RESULT_TSV!
if "!THREADS_SWEEP!"=="1" (
    echo server logs: !ARTIFACT_DIR!\server_t*.log
) else (
    echo server log: !ARTIFACT_DIR!\server_t!THREADS!.log
)
exit /b 0

:run_benchmark
set "THREADS=%~1"
set "SERVER_LOG=!ARTIFACT_DIR!\server_t!THREADS!.log"

call :cleanup_server

echo == launch server (threads=!THREADS!) ==
set "TASK=!PROMPT!"
set "NOISE_MODE=gaussian"
set "NOISE_SEED=-1"
start "model-server-latency-t!THREADS!" /B cmd /c "call ""!LAUNCH_BAT!"" > ""!SERVER_LOG!"" 2>&1"

echo == run latency benchmark (threads=!THREADS!) ==
"%PYTHON%" "!BENCHMARK_SCRIPT!" ^
    --host "!HOST!" ^
    --port !PORT! ^
    !BENCHMARK_IMAGE_ARGS! ^
    --width !IMAGE_WIDTH! ^
    --height !IMAGE_HEIGHT! ^
    --state-dim !STATE_DIM! ^
    --prompt "!PROMPT!" ^
    --warmup !WARMUP! ^
    --loops !LOOPS! ^
    --threads !THREADS! ^
    --wait-server-s !SERVER_WAIT_S! ^
    --result-tsv "!RESULT_TSV!"
if errorlevel 1 exit /b 1

call :cleanup_server
exit /b 0

:cleanup_server
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /C:":!PORT! " ^| findstr LISTENING') do (
    taskkill /F /PID %%P >nul 2>&1
)
exit /b 0
