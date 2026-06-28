# Release Packaging

## Linux

Build redistributable Linux x86_64 packages by selecting one or more variants:

```sh
BUILD_CPU=1 bash tools/release/package_linux.sh
BUILD_CUDA=1 bash tools/release/package_linux.sh
BUILD_CPU=1 BUILD_CUDA=1 bash tools/release/package_linux.sh
```

Outputs:

- `dist/robotcpp-linux-cpu-x86_64.tar.gz`
- `dist/robotcpp-linux-cuda<major>-x86_64.tar.gz`

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
BUILD_CPU=1 JOBS=16 bash tools/release/package_linux.sh
BUILD_CUDA=1 CUDA_MAJOR=12 bash tools/release/package_linux.sh
```

Common knobs:

- `BUILD_CPU`, `BUILD_CUDA`: select package variants.
- `CUDA_MAJOR`: override the detected CUDA package suffix.
- `DIST_DIR`, `BUILD_ROOT`, `JOBS`: control output location and build parallelism.
- `PACKAGE_PREFIX`, `PACKAGE_PLATFORM`, `PACKAGE_ARCH`: override package naming.
- `STRIP_BINARIES`, `SOURCE_PREFIX_MAP`, `CMAKE_BIN`: packaging and toolchain
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
<PACKAGE_PREFIX>-<PACKAGE_PLATFORM>-<variant>-<PACKAGE_ARCH>
```
