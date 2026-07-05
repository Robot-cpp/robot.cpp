# Release Packaging

## Linux

Build redistributable Linux x86_64 packages by selecting one or more variants:

```sh
ROBOT_CPP_BUILD_CPU=1 bash tools/release/package_linux.sh
ROBOT_CPP_BUILD_CUDA=1 bash tools/release/package_linux.sh
ROBOT_CPP_BUILD_CPU=1 ROBOT_CPP_BUILD_CUDA=1 bash tools/release/package_linux.sh
```

Outputs:

- `dist/robot_cpp-linux-cpu-x86_64.tar.gz`
- `dist/robot_cpp-linux-cuda<major>-x86_64.tar.gz`

Package layout:

- `bin/model-server`
- `lib/libggml*.so*`
- `lib/libllama*.so*`

Notes:

- GGUF model files are not included.
- Runtime lookup is package-local through `$ORIGIN`.
- Packaging fails if local source/build paths leak into ELF strings or runtime
  paths.
- CUDA uses the installed toolkit major version in the package name, for
  example `cuda12`.
- CUDA packages do not vendor NVIDIA driver or CUDA runtime libraries.

### Options

Examples:

```sh
ROBOT_CPP_BUILD_CPU=1 ROBOT_CPP_JOBS=16 bash tools/release/package_linux.sh
ROBOT_CPP_BUILD_CUDA=1 ROBOT_CPP_CUDA_MAJOR=12 bash tools/release/package_linux.sh
```

Common knobs:

- `ROBOT_CPP_BUILD_CPU`, `ROBOT_CPP_BUILD_CUDA`: select package variants.
- `ROBOT_CPP_CUDA_MAJOR`: override the detected CUDA package suffix.
- `ROBOT_CPP_DIST_DIR`, `ROBOT_CPP_BUILD_ROOT`, `ROBOT_CPP_JOBS`: control
  output location and build parallelism.
- `ROBOT_CPP_PACKAGE_PREFIX`, `ROBOT_CPP_PACKAGE_PLATFORM`,
  `ROBOT_CPP_PACKAGE_ARCH`: override package naming.
- `ROBOT_CPP_STRIP_BINARIES`, `ROBOT_CPP_SOURCE_PREFIX_MAP`,
  `ROBOT_CPP_CMAKE_BIN`: packaging and toolchain
  controls.

### Adding Variants

Use this shape for new Linux package variants:

1. Add a `build_<variant>_package` function.
2. Configure a dedicated build directory.
3. Build the `model-server` target.
4. Call `stage_package <build_dir> <package_name> <variant>`.

Use `package_name <variant>` to keep package names consistent. The resulting
name is:

```text
<ROBOT_CPP_PACKAGE_PREFIX>-<ROBOT_CPP_PACKAGE_PLATFORM>-<variant>-<ROBOT_CPP_PACKAGE_ARCH>
```
