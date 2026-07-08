# `src` Directory and New Model Guide

[中文](readme_zh.md)

This document describes the core code structure under `robot.cpp/src` and how
to integrate a new robot model runtime into the unified `robotcpp::Model`
abstraction.

## File Structure

```text
src/
├── model-cli.cpp
└── models/
    ├── model.h
    ├── model_factory.cpp
    ├── ggml_backend.cpp
    ├── ggml_backend.h
    ├── gguf_loader.cpp
    ├── gguf_loader.h
    ├── smolvla/
    └── pi0/
```

### `model-cli.cpp`

`model-cli.cpp` is a command-line entrypoint that calls `robotcpp::Model`
directly. It is mainly used for local smoke tests, debugging model arguments,
and comparing model implementations. Its image input interface takes image
paths, which helps keep correctness comparisons reproducible.

#### Build and use `model-cli`

Build only `model-cli`:

```sh
cmake -S . -B build \
  -DROBOT_CPP_BUILD_ROBOT_SERVER=OFF \
  -DROBOT_CPP_BUILD_MODEL_CLI=ON
cmake --build build --target model-cli -j
```

If no backend-related CMake options are passed, the default CPU build is used.
CUDA, Metal, and other backends require explicitly enabling the corresponding
GGML options.

#### SmolVLA single image

Note: `IMAGE_KEY` should match the metadata in the vision GGUF.

```sh
GGUF_DIR=ckpts/smolvla-so101-fp32
IMAGE=examples/main.png
IMAGE_KEY=image
STATE=0,0,0,0,0,0

./build/bin/model-cli \
  --model-type smolvla \
  --llm "${GGUF_DIR}/smolvla-llm-f32.gguf" \
  --mmproj "${GGUF_DIR}/mmproj-smolvla-f32.gguf" \
  --state-proj "${GGUF_DIR}/state-proj-smolvla-f32.gguf" \
  --action-expert "${GGUF_DIR}/action-expert-smolvla-f32.gguf" \
  --image "${IMAGE}" \
  --image-name "${IMAGE_KEY}" \
  --state "${STATE}" \
  --task "grab the block."
```

The `STATE` length must match the checkpoint's state dimension. SO-101
checkpoints are commonly 6-dimensional. `IMAGE_KEY` must match
`smolvla.image_keys` in the GGUF metadata.

#### SmolVLA multiple images

```sh
GGUF_DIR=ckpts/smolvla-libero-fp32
IMAGE0=examples/agentview.png
IMAGE1=examples/eye_in_hand.png
IMAGE_KEY0=observation.images.image
IMAGE_KEY1=observation.images.image2
STATE=0,0,0,0,0,0,0,0

./build/bin/model-cli \
  --model-type smolvla \
  --llm "${GGUF_DIR}/smolvla-llm-f32.gguf" \
  --mmproj "${GGUF_DIR}/mmproj-smolvla-f32.gguf" \
  --state-proj "${GGUF_DIR}/state-proj-smolvla-f32.gguf" \
  --action-expert "${GGUF_DIR}/action-expert-smolvla-f32.gguf" \
  --image "${IMAGE0}" \
  --image "${IMAGE1}" \
  --image-name "${IMAGE_KEY0}" \
  --image-name "${IMAGE_KEY1}" \
  --state "${STATE}" \
  --task "pick up the object."
```

For multiple images, `--image` and `--image-name` are paired by order. Image
keys can differ between checkpoints. Inspect metadata first if needed:

```sh
strings "${GGUF_DIR}/mmproj-smolvla-f32.gguf" | rg "smolvla\.image_keys|observation\.images"
```

#### pi0 multiple images

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/robotcpp-split
MODEL=robotcpp-pi0-libero-finetuned-v044
IMAGE0=examples/agentview.png
IMAGE1=examples/eye_in_hand.png
STATE=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

./build/bin/model-cli \
  --model-type pi0 \
  --vit "${GGUF_DIR}/${MODEL}.vit.gguf" \
  --mmproj "${GGUF_DIR}/${MODEL}.mmproj.gguf" \
  --llm "${GGUF_DIR}/${MODEL}.llm.gguf" \
  --tokenizer "${GGUF_DIR}/${MODEL}.tokenizer.gguf" \
  --state-gguf "${GGUF_DIR}/${MODEL}.state.gguf" \
  --action-decoder "${GGUF_DIR}/${MODEL}.action_decoder.gguf" \
  --image "${IMAGE0}" \
  --image "${IMAGE1}" \
  --image-name observation.images.image \
  --image-name observation.images.image2 \
  --state "${STATE}" \
  --task "pick up the fork"
```

For pi0, `--image-name` must match `pi0.image_keys` in the ViT metadata. The
LIBERO v044 split checkpoint uses `observation.images.image` and
`observation.images.image2`.

### `models/model.h`

`model.h` defines the public model-layer interface. It is the boundary between
`model-cli`, `model-server`, and concrete models.

The main types are:

- `model_type`: supported model enum, such as `smolvla` and `pi0`.
- `model_image`: single image view with name, RGB data pointer, width, height,
  channel count, and stride.
- `observation`: one inference input containing images, robot state vector, and
  task text.
- `model_result`: one inference output containing flattened actions, chunk
  size, action dimension, and optional metrics.
- `model_args`: model initialization arguments, including common parameters and
  model-specific parameters.
- `Model`: unified model base class. The core interface is `predict()`, and the
  optional interface is `reset()`.

Concrete models should wrap their own runtime in a `Model` subclass. External
code should not directly depend on a model's internal engine, cache, or weight
layout.

### `models/model_factory.cpp`

`model_factory.cpp` is the model dispatch entrypoint. `make_model()` creates a
concrete model from `model_args.type`, for example:

```cpp
if (args.type == model_type::smolvla) {
    return make_smolvla_model(args, out, error);
}
if (args.type == model_type::pi0) {
    return make_pi0_model(args, out, error);
}
```

### `models/ggml_backend.*`

These files wrap shared ggml backend, buffer, and scheduler capabilities. They
can be reused by multiple models, avoiding duplicated ggml initialization,
device selection, and memory management in each model directory.

### `models/gguf_loader.*`

These files wrap shared GGUF loading logic. If a model runtime needs to load
split GGUF weights, prefer reusing these helpers instead of reimplementing file
parsing inside the model directory.

### `models/smolvla/`

SmolVLA's implementation directory currently has two main layers:

- `smolvla_engine.*`: model-specific C-style runtime API, such as
  `smolvla_params`, `smolvla_context`, `smolvla_init()`,
  `smolvla_predict_raw_rgb_batch()`, and `smolvla_free()`.
- `smolvla_model.*`: adapts `smolvla_engine` to `robotcpp::Model`. It validates
  `model_args`, initializes the context, converts `observation` to
  `smolvla_image_view`, and converts engine output to `model_result`.

Other files are split by model stage:

- `vision.*`: vision branch.
- `state_proj.*`: state projection.
- `action_expert.*`: action expert.
- `smolvla_compat.h`, `smolvla_llama_compat.h`: compatibility layers for
  dependency versions or llama.cpp.

### `models/pi0/`

pi0 also follows the "model-specific engine + `Model` subclass" structure:

- `pi0_engine.*`: model-specific runtime API, such as `pi0_params`,
  `pi0_context`, `pi0_init()`, `pi0_predict_raw_rgb()`, `pi0_reset()`, and
  `pi0_free()`.
- `pi0_model.*`: adapts `pi0_engine` to `robotcpp::Model`.

Other files are split by inference stage:

- `load.*`, `weights.*`: load model components and weights.
- `preprocess.*`: preprocess input images and state.
- `vlm.*`: vision-language model stages.
- `action.*`: action decoding or sampling logic.
- `component_runtime.*`, `pi0_context.h`, `config.h`, `types.h`: component
  runtime, context, and configuration types.

## Inference Call Path

The core call path is:

```text
model-cli / model-server
        |
        v
robotcpp::make_model(model_args)
        |
        v
Concrete Model wrapper, e.g. SmolVLAModel / Pi0Model
        |
        v
Model-specific engine, e.g. smolvla_engine / pi0_engine
        |
        v
model_result(actions, chunk_size, action_dim, metrics)
```

`model-cli` and `model-server` should not understand each model's internals.
They only prepare `observation`, select `model_type`, pass `model_args`, and
consume the unified `model_result`.

## How to Add a New Model

Assume the new model is named `newmodel`. The recommended integration flow is
below.

### 0. Add conversion and quantization tooling

Before adding a C++ runtime, it is recommended to add checkpoint-to-GGUF
conversion first.

Add two pieces:

- `tools/hf2gguf/<model_name>/`: add a converter such as
  `convert_<model_name>_to_gguf.py`, or split it by component, to convert the
  original checkpoint or intermediate `.pt` files into GGUF files loaded by the
  C++ runtime. If the original checkpoint must first be split into components,
  refer to `tools/hf2gguf/smolvla/smolvla_surgery.py`: emit components such as
  vision, state, and action `.pt` files, then convert each one to GGUF. If the
  flow has many steps, add a `convert_<model_name>_all.sh` wrapper.
- `tools/quant/config/<model_name>_origin.yaml`: add a quant plan listing each
  GGUF component's `input`, `output`, and tensor groups. Every tensor should be
  matched by exactly one group. Tensors that should not be quantized, such as
  norm, bias, embedding, normalizer, or unnormalizer tensors, should be clearly
  marked with `quantizable: false`.

### 1. Create the model directory

Create a directory under `src/models/`:

```text
src/models/newmodel/
├── newmodel_model.cpp
└── newmodel_model.h
```

`newmodel_model.*` is the unified entrypoint that implements
`robotcpp::Model`.

For the implementation itself, we recommend splitting the model into modules.
This gives Robot.cpp more flexibility and composability, because different
architectures in different modules can have different optimal behavior. In
practice, split modules such as vision, state, LLM, and action for both loading
and compute. This mirrors the outer `robotcpp::Model`: initialization loads all
GGUFs, and prediction runs one forward pass; each smaller component similarly
loads once during init and computes once during predict.

### 2. Implement the `Model` subclass

Add `newmodel_model.h` and define a class that inherits `robotcpp::Model`:

```cpp
#pragma once

#include "models/model.h"

#include <memory>
#include <string>

namespace robotcpp {

class NewModel final : public Model {
  public:
    explicit NewModel(const model_args & args);
    ~NewModel() override;

    NewModel(const NewModel &) = delete;
    NewModel & operator=(const NewModel &) = delete;

    const char * type() const override;
    bool predict(const observation & obs, model_result & out, std::string & error) override;
    void reset() override;

    bool is_ready() const;

  private:
    model_args args_;

    // Store runtime, weights, cache, or component objects as needed.
};

bool make_newmodel(const model_args & args, std::unique_ptr<Model> & out, std::string & error);

} // namespace robotcpp
```

Implement `newmodel_model.cpp` according to the responsibilities of
`NewModel`:

- `NewModel(const model_args & args)`: save initialization arguments, validate
  required weight paths or configuration, and initialize the model's internal
  runtime, weights, cache, or component objects.
- `~NewModel()`: release resources created by the constructor, ensuring no
  context, backend buffer, or external runtime handle leaks when the object is
  destroyed.
- `type()`: return a stable model type string, such as `"newmodel"`, for logs
  and debugging.
- `predict()`: validate images, state, and task in `observation`, run one model
  inference, and convert the internal result to `model_result`.
- `reset()`: clear cache, temporal state, or reused sampling state if the model
  has any. Stateless models can leave it empty.
- `is_ready()`: return whether initialization succeeded, so the factory can
  check after creation.
- `make_newmodel()`: create a `NewModel` instance, handle allocation or
  initialization failure, and return a readable message through `error`.

#### Reuse GGUF/ggml initialization

Initialization often needs ggml backend setup and GGUF loading. These flows are
usually highly similar across models, so use the shared helpers directly.

For GGUF loading, each model should usually derive from the GGUF loader base and
override tensor binding and metadata parsing, because those two steps tend to be
model-specific.

For ggml backends, use the shared backend wrapper, which handles backend
initialization, scheduler initialization, and buffer-type policy.

### 3. Extend `model_args` and `model_type`

In `src/models/model.h`:

1. Add a new enum value to `model_type`, such as `newmodel`.
2. Add any parameters required by the new model to `model_args`, such as
   `std::string newmodel_path;`.

If existing fields such as `llm_path`, `mmproj_path`, or `state_path` match the
semantics, reuse them. If the meaning differs, add clearly named fields to avoid
maintenance confusion later. The initialization skeleton above uses
`args_.newmodel_path` to represent the new weight path field added in this step.

### 4. Wire into the factory

Include the new model subclass and add dispatch in
`src/models/model_factory.cpp`:

```cpp
#include "models/newmodel/newmodel_model.h"

bool make_model(const model_args & args, std::unique_ptr<Model> & out, std::string & error) {
    out.reset();
    if (args.type == model_type::newmodel) {
        return make_newmodel(args, out, error);
    }
    ...
}
```

After this, `model-cli` and `model-server` can create the new model through the
unified entrypoint.

### 5. Wire into `model-cli`

At minimum, update `src/model-cli.cpp`:

- `parse_model_type()`: recognize the new `--model-type newmodel`.
- `print_usage()`: document the new model arguments.
- Argument parsing loop: parse the CLI arguments required by the new model and
  write them into `model_args`.

If the new model has special requirements for image names, state dimensions, or
task text, return clear errors from the subclass's `predict()` or initialization
validation so both CLI and server paths can reuse them.

### 6. Update CMake

Add the new model files to the appropriate target in `CMakeLists.txt`. The
minimal integration only needs `robotcpp` to compile and link
`newmodel_model.*` and its direct internal dependencies:

```cmake
add_library(robotcpp STATIC
    src/models/model.h
    src/models/model_factory.cpp
    src/models/newmodel/newmodel_model.cpp
    src/models/newmodel/newmodel_model.h
    ...
)
```

### 7. Verify

#### Minimal verification template for a new model

After integrating the new model, first run a minimal inference with
`model-cli`:

```sh
./build/bin/model-cli \
  --model-type newmodel \
  --image /path/to/image.png \
  --image-name image \
  --state 0,0,0,0,0,0 \
  --task "grab the block." \
  --threads 4 \
  --newmodel-weights /path/to/newmodel.gguf
```

If `model-server` should also support the model, confirm that server startup
arguments can pass the new model's required fields into `model_args`.

## New Model Checklist

- `tools/hf2gguf/<model_name>/` can convert the original checkpoint or `.pt`
  files into GGUF files used by the C++ runtime.
- `tools/quant/config/<model_name>_origin.yaml` covers every GGUF component and
  tensor group for the new model.
- `src/models/<model_name>/` contains a clear `Model` subclass. Only split out
  an internal runtime or engine when it is genuinely needed.
- `model_type` has a new enum value.
- `model_args` contains the parameters required for new model initialization.
- `model_factory.cpp` dispatches to the new model.
- `model-cli.cpp` supports `--model-type <model_name>` and required arguments.
- `CMakeLists.txt` includes the new model source files and lets `robotcpp` link
  against the required internal components.
