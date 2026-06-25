@echo off
setlocal enabledelayedexpansion

rem Configure + build + launch model-server on Windows (MinGW, CPU only).
rem Run from PowerShell/CMD, or MSYS2 UCRT64 with gcc/cmake/mingw32-make on PATH.
rem
rem Usage:
rem   set ROBOT_CPP_ROOT=C:\path\to\robot.cpp
rem   set GGUF_DIR=C:\path\to\robot.cpp\ckpts\smolvla_grab_block_50_20k_f16
rem   set DTYPE=f16
rem   robot_server\shell\launch_robot_server_windows_cpu.bat
rem
rem Optional:
rem   set MINGW_BIN=C:\msys64\ucrt64\bin
rem   set SKIP_BUILD=1
rem
rem MSYS2 packages (UCRT64 shell):
rem   pacman -S --needed mingw-w64-ucrt-x86_64-toolchain

if not defined ROBOT_CPP_ROOT (
    echo error: ROBOT_CPP_ROOT must be set. >&2
    exit /b 1
)

if not defined MODEL_TYPE set "MODEL_TYPE=smolvla"
if not defined DTYPE set "DTYPE=f16"
if not defined BUILD_DIR set "BUILD_DIR=!ROBOT_CPP_ROOT!\build-win-cpu-mingw"

if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=5555"
if not defined THREADS set "THREADS=8"
if not defined TASK set "TASK=grab the block."
if not defined N_BATCH set "N_BATCH=512"
if not defined N_CTX set "N_CTX=2048"
if not defined NOISE_MODE set "NOISE_MODE=gaussian"
if not defined NOISE_SEED set "NOISE_SEED=-1"
if not defined VERBOSITY set "VERBOSITY=0"

if not defined SKIP_BUILD set "SKIP_BUILD=0"
if not defined CMAKE_BIN set "CMAKE_BIN=cmake"
if not defined CMAKE_GENERATOR set "CMAKE_GENERATOR=MinGW Makefiles"
if not defined CMAKE_BUILD_TYPE set "CMAKE_BUILD_TYPE=Release"
if not defined GGML_NATIVE set "GGML_NATIVE=OFF"
if not defined GGML_BLAS set "GGML_BLAS=OFF"
if not defined GGML_OPENMP set "GGML_OPENMP=OFF"

if not defined CC set "CC=gcc"
if not defined CXX set "CXX=g++"

if not defined MINGW_BIN (
    if exist "C:\msys64\ucrt64\bin\gcc.exe" set "MINGW_BIN=C:\msys64\ucrt64\bin"
    if not defined MINGW_BIN if exist "C:\msys64\mingw64\bin\gcc.exe" set "MINGW_BIN=C:\msys64\mingw64\bin"
)
if defined MINGW_BIN (
    set "PATH=!MINGW_BIN!;!PATH!"
    if not defined CMAKE_MAKE_PROGRAM if exist "!MINGW_BIN!\mingw32-make.exe" set "CMAKE_MAKE_PROGRAM=!MINGW_BIN!\mingw32-make.exe"
)

where gcc >nul 2>&1
if errorlevel 1 (
    echo error: gcc not found. Install MSYS2 UCRT64 toolchain and add it to PATH, or set MINGW_BIN. >&2
    exit /b 1
)

where mingw32-make >nul 2>&1
if errorlevel 1 (
    echo error: mingw32-make not found. Install mingw-w64-ucrt-x86_64-make in MSYS2 UCRT64. >&2
    exit /b 1
)

if not defined GGUF_DIR (
    if /I "!MODEL_TYPE!"=="smolvla" (
        set "GGUF_DIR=!ROBOT_CPP_ROOT!\ckpts\smolvla\gguf-!DTYPE!"
    ) else if /I "!MODEL_TYPE!"=="pi0" (
        set "GGUF_DIR=!ROBOT_CPP_ROOT!\ckpts\pi0-libero-finetuned-v044\robotcpp-split"
    ) else (
        echo unsupported MODEL_TYPE=!MODEL_TYPE! >&2
        exit /b 1
    )
)

if not "!SKIP_BUILD!"=="1" (
    echo == configure ==
    echo generator: !CMAKE_GENERATOR!
    echo build_type: !CMAKE_BUILD_TYPE!
    echo cc: !CC!
    echo cxx: !CXX!
    if defined CMAKE_MAKE_PROGRAM echo make: !CMAKE_MAKE_PROGRAM!

    if defined CMAKE_MAKE_PROGRAM (
        if defined GGML_BLAS_VENDOR (
            "%CMAKE_BIN%" -S "!ROBOT_CPP_ROOT!" -B "!BUILD_DIR!" -G "!CMAKE_GENERATOR!" ^
                -DCMAKE_BUILD_TYPE=!CMAKE_BUILD_TYPE! ^
                -DCMAKE_C_COMPILER=!CC! ^
                -DCMAKE_CXX_COMPILER=!CXX! ^
                -DCMAKE_MAKE_PROGRAM="!CMAKE_MAKE_PROGRAM!" ^
                -DGGML_NATIVE=!GGML_NATIVE! ^
                -DGGML_BLAS=!GGML_BLAS! ^
                -DGGML_BLAS_VENDOR=!GGML_BLAS_VENDOR! ^
                -DGGML_OPENMP=!GGML_OPENMP! ^
                -DGGML_CUDA=OFF ^
                -DGGML_METAL=OFF ^
                -DROBOT_CPP_BUILD_ROBOT_SERVER=ON
        ) else (
            "%CMAKE_BIN%" -S "!ROBOT_CPP_ROOT!" -B "!BUILD_DIR!" -G "!CMAKE_GENERATOR!" ^
                -DCMAKE_BUILD_TYPE=!CMAKE_BUILD_TYPE! ^
                -DCMAKE_C_COMPILER=!CC! ^
                -DCMAKE_CXX_COMPILER=!CXX! ^
                -DCMAKE_MAKE_PROGRAM="!CMAKE_MAKE_PROGRAM!" ^
                -DGGML_NATIVE=!GGML_NATIVE! ^
                -DGGML_BLAS=!GGML_BLAS! ^
                -DGGML_OPENMP=!GGML_OPENMP! ^
                -DGGML_CUDA=OFF ^
                -DGGML_METAL=OFF ^
                -DROBOT_CPP_BUILD_ROBOT_SERVER=ON
        )
    ) else (
        if defined GGML_BLAS_VENDOR (
            "%CMAKE_BIN%" -S "!ROBOT_CPP_ROOT!" -B "!BUILD_DIR!" -G "!CMAKE_GENERATOR!" ^
                -DCMAKE_BUILD_TYPE=!CMAKE_BUILD_TYPE! ^
                -DCMAKE_C_COMPILER=!CC! ^
                -DCMAKE_CXX_COMPILER=!CXX! ^
                -DGGML_NATIVE=!GGML_NATIVE! ^
                -DGGML_BLAS=!GGML_BLAS! ^
                -DGGML_BLAS_VENDOR=!GGML_BLAS_VENDOR! ^
                -DGGML_OPENMP=!GGML_OPENMP! ^
                -DGGML_CUDA=OFF ^
                -DGGML_METAL=OFF ^
                -DROBOT_CPP_BUILD_ROBOT_SERVER=ON
        ) else (
            "%CMAKE_BIN%" -S "!ROBOT_CPP_ROOT!" -B "!BUILD_DIR!" -G "!CMAKE_GENERATOR!" ^
                -DCMAKE_BUILD_TYPE=!CMAKE_BUILD_TYPE! ^
                -DCMAKE_C_COMPILER=!CC! ^
                -DCMAKE_CXX_COMPILER=!CXX! ^
                -DGGML_NATIVE=!GGML_NATIVE! ^
                -DGGML_BLAS=!GGML_BLAS! ^
                -DGGML_OPENMP=!GGML_OPENMP! ^
                -DGGML_CUDA=OFF ^
                -DGGML_METAL=OFF ^
                -DROBOT_CPP_BUILD_ROBOT_SERVER=ON
        )
    )
    if errorlevel 1 exit /b 1

    echo == build ==
    "%CMAKE_BIN%" --build "!BUILD_DIR!" --target model-server -j 8
    if errorlevel 1 exit /b 1
)

set "SERVER_BIN=!BUILD_DIR!\bin\model-server.exe"
if not exist "!SERVER_BIN!" set "SERVER_BIN=!BUILD_DIR!\bin\Release\model-server.exe"
if not exist "!SERVER_BIN!" (
    echo error: model-server.exe not found under "!BUILD_DIR!\bin" >&2
    exit /b 1
)

echo == launch server ==
echo model_type: !MODEL_TYPE!
echo host: !HOST!
echo port: !PORT!
echo gguf_dir: !GGUF_DIR!
echo task: !TASK!
echo server_bin: !SERVER_BIN!

if /I "!MODEL_TYPE!"=="smolvla" (
    if not defined LLM_GGUF set "LLM_GGUF=!GGUF_DIR!\smolvla-llm-!DTYPE!.gguf"
    if not defined VISION_GGUF set "VISION_GGUF=!GGUF_DIR!\mmproj-smolvla-!DTYPE!.gguf"
    if not defined STATE_PROJ_GGUF set "STATE_PROJ_GGUF=!GGUF_DIR!\state-proj-smolvla-!DTYPE!.gguf"
    if not defined ACTION_EXPERT_GGUF set "ACTION_EXPERT_GGUF=!GGUF_DIR!\action-expert-smolvla-!DTYPE!.gguf"
    "!SERVER_BIN!" ^
        --model-type smolvla ^
        --llm "!LLM_GGUF!" ^
        --mmproj "!VISION_GGUF!" ^
        --state-proj "!STATE_PROJ_GGUF!" ^
        --action-expert "!ACTION_EXPERT_GGUF!" ^
        --task "!TASK!" ^
        --host "!HOST!" ^
        --port !PORT! ^
        --threads !THREADS! ^
        --n-batch !N_BATCH! ^
        --n-ctx !N_CTX! ^
        --noise-mode !NOISE_MODE! ^
        --noise-seed !NOISE_SEED! ^
        --verbosity !VERBOSITY!
    exit /b !ERRORLEVEL!
)

if /I "!MODEL_TYPE!"=="pi0" (
    if not defined ROBOTCPP_BACKEND set "ROBOTCPP_BACKEND=cpu"
    if not defined MODEL_BASENAME set "MODEL_BASENAME=robotcpp-pi0-libero-finetuned-v044"
    if not defined VIT_GGUF set "VIT_GGUF=!GGUF_DIR!\!MODEL_BASENAME!.vit.gguf"
    if not defined MMPROJ_GGUF set "MMPROJ_GGUF=!GGUF_DIR!\!MODEL_BASENAME!.mmproj.gguf"
    if not defined LLM_GGUF set "LLM_GGUF=!GGUF_DIR!\!MODEL_BASENAME!.llm.gguf"
    if not defined TOKENIZER_GGUF set "TOKENIZER_GGUF=!GGUF_DIR!\!MODEL_BASENAME!.tokenizer.gguf"
    if not defined STATE_GGUF set "STATE_GGUF=!GGUF_DIR!\!MODEL_BASENAME!.state.gguf"
    if not defined ACTION_DECODER_GGUF set "ACTION_DECODER_GGUF=!GGUF_DIR!\!MODEL_BASENAME!.action_decoder.gguf"
    "!SERVER_BIN!" ^
        --model-type pi0 ^
        --vit "!VIT_GGUF!" ^
        --mmproj "!MMPROJ_GGUF!" ^
        --llm "!LLM_GGUF!" ^
        --tokenizer "!TOKENIZER_GGUF!" ^
        --state-gguf "!STATE_GGUF!" ^
        --action-decoder "!ACTION_DECODER_GGUF!" ^
        --task "!TASK!" ^
        --host "!HOST!" ^
        --port !PORT! ^
        --threads !THREADS! ^
        --n-batch !N_BATCH! ^
        --n-ctx !N_CTX! ^
        --noise-mode !NOISE_MODE! ^
        --noise-seed !NOISE_SEED! ^
        --verbosity !VERBOSITY!
    exit /b !ERRORLEVEL!
)

echo unsupported MODEL_TYPE=!MODEL_TYPE! >&2
exit /b 1
