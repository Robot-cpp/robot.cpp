@echo off
setlocal enabledelayedexpansion

rem Configure + build + launch model-server on Windows (MSVC + CUDA).
rem Requires Visual Studio 2022 (Desktop C++) and NVIDIA CUDA Toolkit.
rem Default generator is Ninja (recommended). Visual Studio generator needs CUDA VS integration.
rem Run from "Developer PowerShell for VS 2022" or "x64 Native Tools Command Prompt".
rem
rem Usage:
rem   set ROBOT_CPP_ROOT=C:\path\to\robot.cpp
rem   set GGUF_DIR=C:\path\to\robot.cpp\ckpts\smolvla_grab_block_50_20k_f16
rem   set CMAKE_CUDA_ARCHITECTURES=86
rem   robot_server\shell\launch_robot_server_windows_cuda.bat
rem
rem Optional:
rem   set SKIP_BUILD=1
rem   set CMAKE_GENERATOR=Visual Studio 17 2022
rem   set VSROOT=C:\Program Files\Microsoft Visual Studio\2022\Community

if not defined ROBOT_CPP_ROOT (
    echo error: ROBOT_CPP_ROOT must be set. >&2
    exit /b 1
)

if not defined MODEL_TYPE set "MODEL_TYPE=smolvla"
if not defined DTYPE set "DTYPE=f16"
if not defined BUILD_DIR set "BUILD_DIR=!ROBOT_CPP_ROOT!\build-win-cuda-msvc"
if not defined CMAKE_CONFIG set "CMAKE_CONFIG=Release"

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
if not defined CMAKE_GENERATOR set "CMAKE_GENERATOR=Ninja"
if not defined CMAKE_PLATFORM set "CMAKE_PLATFORM=x64"
if not defined CMAKE_BUILD_TYPE set "CMAKE_BUILD_TYPE=Release"
if not defined GGML_NATIVE set "GGML_NATIVE=OFF"
if not defined GGML_OPENMP set "GGML_OPENMP=OFF"

if not defined CUDA_PATH if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
)
if defined CUDA_PATH (
    if not defined CMAKE_CUDA_COMPILER set "CMAKE_CUDA_COMPILER=!CUDA_PATH!\bin\nvcc.exe"
    set "PATH=!CUDA_PATH!\bin;!PATH!"
)

if not defined VSROOT (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq delims=" %%I in (`"!VSWHERE!" -latest -property installationPath`) do set "VSROOT=%%I"
    )
)

rem Always load x64 MSVC toolchain. Regular PowerShell may expose Hostx86\x86 cl.exe first,
rem which breaks nvcc/cudafe++ (ACCESS_VIOLATION during CUDA compiler detection).
if defined VSROOT (
    if exist "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" (
        echo == load x64 MSVC environment ==
        call "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if not defined VCToolsInstallDir (
    echo error: VCToolsInstallDir not set after vcvars64. Use VS 2022 x64 developer tools. >&2
    exit /b 1
)
set "MSVC_CL=!VCToolsInstallDir!bin\Hostx64\x64\cl.exe"
if not exist "!MSVC_CL!" (
    echo error: cl.exe not found: !MSVC_CL! >&2
    exit /b 1
)
echo msvc_cl: !MSVC_CL!

rem nvcc -ccbin breaks on quoted MSVC paths with spaces during CMake CUDA detection.
rem After vcvars64, "cl" on PATH is the correct x64 host compiler.
set "CUDA_HOST_COMPILER=cl"

if not defined CMAKE_MAKE_PROGRAM (
    if defined VSROOT if exist "!VSROOT!\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" (
        set "CMAKE_MAKE_PROGRAM=!VSROOT!\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
    )
)

where nvcc >nul 2>&1
if errorlevel 1 (
    echo error: nvcc not found. Install NVIDIA CUDA Toolkit and add it to PATH. >&2
    exit /b 1
)

if /I "!CMAKE_GENERATOR!"=="Visual Studio 17 2022" (
    if defined VSROOT (
        set "VS_CUDA_PROPS=!VSROOT!\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.9.props"
        if not exist "!VS_CUDA_PROPS!" (
            echo warning: CUDA Visual Studio integration not found at "!VS_CUDA_PROPS!" >&2
            echo         Copy files from "%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions" >&2
            echo         into "!VSROOT!\MSBuild\Microsoft\VC\v170\BuildCustomizations\" (admin), >&2
            echo         or set CMAKE_GENERATOR=Ninja to skip VS CUDA toolset. >&2
        )
    )
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
    echo build_dir: !BUILD_DIR!
    echo generator: !CMAKE_GENERATOR!
    echo config: !CMAKE_CONFIG!
    cl 2>nul | findstr /C:"Version" | findstr /V "findstr"
    nvcc --version | findstr /C:"release"

    set "COMMON_CUDA_ARCH="
    if defined CMAKE_CUDA_ARCHITECTURES set "COMMON_CUDA_ARCH=-DCMAKE_CUDA_ARCHITECTURES=!CMAKE_CUDA_ARCHITECTURES!"

    if /I "!CMAKE_GENERATOR!"=="Ninja" (
        echo build_type: !CMAKE_BUILD_TYPE!
        if defined CMAKE_MAKE_PROGRAM (
            "%CMAKE_BIN%" -S "!ROBOT_CPP_ROOT!" -B "!BUILD_DIR!" -G Ninja ^
                -DCMAKE_BUILD_TYPE=!CMAKE_BUILD_TYPE! ^
                -DCMAKE_C_COMPILER="!MSVC_CL!" ^
                -DCMAKE_CXX_COMPILER="!MSVC_CL!" ^
                -DCMAKE_CUDA_HOST_COMPILER=!CUDA_HOST_COMPILER! ^
                -DCMAKE_MAKE_PROGRAM="!CMAKE_MAKE_PROGRAM!" ^
                -DCMAKE_CUDA_COMPILER="!CMAKE_CUDA_COMPILER!" ^
                -DGGML_NATIVE=!GGML_NATIVE! ^
                -DGGML_OPENMP=!GGML_OPENMP! ^
                -DGGML_CUDA=ON ^
                -DGGML_METAL=OFF ^
                -DBUILD_SHARED_LIBS=OFF ^
                -DROBOT_CPP_BUILD_ROBOT_SERVER=ON ^
                !COMMON_CUDA_ARCH!
        ) else (
            "%CMAKE_BIN%" -S "!ROBOT_CPP_ROOT!" -B "!BUILD_DIR!" -G Ninja ^
                -DCMAKE_BUILD_TYPE=!CMAKE_BUILD_TYPE! ^
                -DCMAKE_C_COMPILER="!MSVC_CL!" ^
                -DCMAKE_CXX_COMPILER="!MSVC_CL!" ^
                -DCMAKE_CUDA_HOST_COMPILER=!CUDA_HOST_COMPILER! ^
                -DCMAKE_CUDA_COMPILER="!CMAKE_CUDA_COMPILER!" ^
                -DGGML_NATIVE=!GGML_NATIVE! ^
                -DGGML_OPENMP=!GGML_OPENMP! ^
                -DGGML_CUDA=ON ^
                -DGGML_METAL=OFF ^
                -DBUILD_SHARED_LIBS=OFF ^
                -DROBOT_CPP_BUILD_ROBOT_SERVER=ON ^
                !COMMON_CUDA_ARCH!
        )
    ) else (
        echo platform: !CMAKE_PLATFORM!
        if defined CMAKE_CUDA_ARCHITECTURES (
            "%CMAKE_BIN%" -S "!ROBOT_CPP_ROOT!" -B "!BUILD_DIR!" -G "!CMAKE_GENERATOR!" -A "!CMAKE_PLATFORM!" ^
                -T cuda="!CUDA_PATH!" ^
                -DGGML_NATIVE=!GGML_NATIVE! ^
                -DGGML_OPENMP=!GGML_OPENMP! ^
                -DGGML_CUDA=ON ^
                -DGGML_METAL=OFF ^
                -DBUILD_SHARED_LIBS=OFF ^
                -DROBOT_CPP_BUILD_ROBOT_SERVER=ON ^
                -DCMAKE_CUDA_ARCHITECTURES=!CMAKE_CUDA_ARCHITECTURES!
        ) else (
            "%CMAKE_BIN%" -S "!ROBOT_CPP_ROOT!" -B "!BUILD_DIR!" -G "!CMAKE_GENERATOR!" -A "!CMAKE_PLATFORM!" ^
                -T cuda="!CUDA_PATH!" ^
                -DGGML_NATIVE=!GGML_NATIVE! ^
                -DGGML_OPENMP=!GGML_OPENMP! ^
                -DGGML_CUDA=ON ^
                -DGGML_METAL=OFF ^
                -DBUILD_SHARED_LIBS=OFF ^
                -DROBOT_CPP_BUILD_ROBOT_SERVER=ON
        )
    )
    if errorlevel 1 exit /b 1

    echo == build ==
    if /I "!CMAKE_GENERATOR!"=="Ninja" (
        "%CMAKE_BIN%" --build "!BUILD_DIR!" --target model-server -j 8
    ) else (
        "%CMAKE_BIN%" --build "!BUILD_DIR!" --config !CMAKE_CONFIG! --target model-server -j 8
    )
    if errorlevel 1 exit /b 1
)

set "SERVER_BIN=!BUILD_DIR!\bin\!CMAKE_CONFIG!\model-server.exe"
if not exist "!SERVER_BIN!" set "SERVER_BIN=!BUILD_DIR!\bin\model-server.exe"
if not exist "!SERVER_BIN!" set "SERVER_BIN=!BUILD_DIR!\!CMAKE_CONFIG!\bin\model-server.exe"
if not exist "!SERVER_BIN!" (
    echo error: model-server.exe not found under "!BUILD_DIR!" >&2
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
