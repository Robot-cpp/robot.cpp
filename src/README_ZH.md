# `src` 目录说明与新增模型指南

这份文档说明 `robot.cpp/src` 下的核心代码结构，以及如何把一个新的机器人模型 runtime 接入到统一的 `robotcpp::Model` 抽象中。

## 文件结构

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

`model-cli.cpp` 是一个直接调用 `robotcpp::Model` 的命令行入口，主要用于本地 smoke test、调试模型参数和进行模型实现的对拍。因此其image的输入接口为传入图片路径，以保证对拍的一致性。

#### 编译和使用 `model-cli`

只编译 `model-cli`：

```sh
cmake -S . -B build \
  -DROBOT_CPP_BUILD_ROBOT_SERVER=OFF \
  -DROBOT_CPP_BUILD_MODEL_CLI=ON
cmake --build build --target model-cli -j
```

不传 backend 相关 CMake 选项时走默认 CPU 编译；CUDA、Metal 等后端需要显式打开对应的 GGML 选项。

#### SmolVLA 单图

注意：IMAGE_KEY应当参考gguf中的vision部分的meta data

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

`STATE` 的长度需要和 checkpoint 的 state 维度一致；SO-101 常见是 6 维。`IMAGE_KEY` 需要和 GGUF metadata 里的 `smolvla.image_keys` 对上。

#### SmolVLA 多图

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

多图时，`--image` 和 `--image-name` 按顺序配对。不同 checkpoint 的 image key 可能不同，可以先查 metadata：

```sh
strings "${GGUF_DIR}/mmproj-smolvla-f32.gguf" | rg "smolvla\.image_keys|observation\.images"
```

#### pi0 多图

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

pi0 的 `--image-name` 必须和vit部分的metadata 里的 `pi0.image_keys` 一致。LIBERO v044 split checkpoint 使用 `observation.images.image` 和 `observation.images.image2`。

### `models/model.h`

`model.h` 定义了模型层的公共接口，是 `model-cli`、`model-server` 和具体模型之间的边界。

主要类型包括：

* `model_type`：当前支持的模型枚举，例如 `smolvla`、`pi0`。
* `model_image`：单张图像视图，包含名字、RGB 数据指针、宽高、通道数和 stride。
* `observation`：一次推理输入，包含多张图像、机器人状态向量和任务文本。
* `model_result`：一次推理输出，包含扁平化 action、chunk size、action dim 和可选 metrics。
* `model_args`：模型初始化参数，包含通用参数和各模型专用参数。
* `Model`：统一模型基类，核心接口是 `predict()`，可选接口是 `reset()`。

具体模型需要把自己的 runtime 封装成 `Model` 子类，外部不直接依赖模型内部的 engine、cache 或权重结构。

### `models/model_factory.cpp`

`model_factory.cpp` 是模型分发入口。`make_model()` 根据 `model_args.type` 创建具体模型，例如：

```cpp
if (args.type == model_type::smolvla) {
    return make_smolvla_model(args, out, error);
}
if (args.type == model_type::pi0) {
    return make_pi0_model(args, out, error);
}
```

### `models/ggml_backend.*`

这组文件封装 ggml 后端、buffer、scheduler 等公共能力。多个模型都可能复用这层代码，避免每个模型目录里重复处理 ggml 初始化、设备选择和内存管理。

### `models/gguf_loader.*`

这组文件封装 GGUF 读取相关的公共逻辑。模型 runtime 如果需要读取 split GGUF 权重，优先复用这里的 helper，而不是在模型目录里重新写一套文件解析逻辑。

### `models/smolvla/`

SmolVLA 的模型实现目录。当前主要分为两层：

* `smolvla_engine.*`：模型专用的 C 风格 runtime API，例如 `smolvla_params`、`smolvla_context`、`smolvla_init()`、`smolvla_predict_raw_rgb_batch()`、`smolvla_free()`。
* `smolvla_model.*`：把 `smolvla_engine` 适配到 `robotcpp::Model`，负责校验 `model_args`、初始化 context、把 `observation` 转成 `smolvla_image_view`，并把 engine 输出转成 `model_result`。

其他文件按模型内部阶段拆分，例如：

* `vision.*`：视觉分支。
* `state_proj.*`：状态投影。
* `action_expert.*`：action expert。
* `smolvla_compat.h`、`smolvla_llama_compat.h`：与依赖版本或 llama.cpp 相关的兼容层。

### `models/pi0/`

pi0 的模型实现目录，也采用“模型专用 engine + `Model` 子类”的结构：

* `pi0_engine.*`：模型专用 runtime API，例如 `pi0_params`、`pi0_context`、`pi0_init()`、`pi0_predict_raw_rgb()`、`pi0_reset()`、`pi0_free()`。
* `pi0_model.*`：把 `pi0_engine` 适配到 `robotcpp::Model`。

其他文件按推理流程拆分，例如：

* `load.*`、`weights.*`：加载模型组件和权重。
* `preprocess.*`：输入图像和状态预处理。
* `vlm.*`：视觉语言模型相关阶段。
* `action.*`：action 解码或采样逻辑。
* `component_runtime.*`、`pi0_context.h`、`config.h`、`types.h`：组件运行时、上下文和配置类型。

## 推理调用路径

当前核心调用路径可以理解为：

```text
model-cli / model-server
        |
        v
robotcpp::make_model(model_args)
        |
        v
具体 Model wrapper，例如 SmolVLAModel / Pi0Model
        |
        v
模型专用 engine，例如 smolvla_engine / pi0_engine
        |
        v
model_result(actions, chunk_size, action_dim, metrics)
```

`model-cli` 和 `model-server` 不应该理解每个模型的内部结构。它们只负责准备 `observation`、选择 `model_type`、传入 `model_args`，然后消费统一的 `model_result`。

## 如何新增一个新的 model

假设新模型名为 `newmodel`，建议按下面的顺序接入。

### 0. 补齐 tools 转换和量化链路

新增 C++ runtime 之前，建议先把 checkpoint 到 GGUF 的转换补上。

需要补两块内容：

* `tools/hf2gguf/<model_name>/`：新增 converter，例如 `convert_<model_name>_to_gguf.py` 或按组件拆成多个 converter，把原始 checkpoint 或中间 `.pt` 文件转换成 C++ runtime 会加载的 GGUF 文件。如果原始 checkpoint 需要先拆成多个组件，可以参考 `tools/hf2gguf/smolvla/smolvla_surgery.py`，先输出如 vision、state、action 等 `.pt`，再分别转成 GGUF；如果步骤较多，再补一个 `convert_<model_name>_all.sh` 串起完整流程。
* `tools/quant/config/<model_name>_origin.yaml`：新增 quant plan，列出新模型每个 GGUF component 的 `input`、`output` 和 tensor 分组。每个 tensor 应该被且只被一个 group 匹配；norm、bias、embedding、normalizer/unnormalizer 等不适合量化的 tensor 应明确标成 `quantizable: false`

### 1. 新建模型目录

在 `src/models/` 下新建目录：

```text
src/models/newmodel/
├── newmodel_model.cpp
└── newmodel_model.h
```

`newmodel_model.*` 是统一入口，用来实现 `robotcpp::Model`

具体的实现中，我们推荐分模块进行实现，这给robot.cpp带来了更大的灵活性与组合性，因为不同模块的不同架构，可能导致不同模块的最优行为可能有差异。因此我们实际上推荐分模块（譬如，vision部分，state部分，llm部分，action部分）来进行load和compute，这像是一种分形，最外层的`robotcpp::Model`也是有init和predict负责load所有的gguf，以及运行一次forward；在init中，每一个小组件完成一次load，在predict中，每一个小组件亦都完成一次compute

### 2. 实现 `Model` 子类

新增 `newmodel_model.h`，定义一个继承 `robotcpp::Model` 的类：

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

    // 按模型需要保存 runtime、权重、cache 或组件对象。
};

bool make_newmodel(const model_args & args, std::unique_ptr<Model> & out, std::string & error);

} // namespace robotcpp
```

`newmodel_model.cpp` 建议按 `NewModel` 类的职责来实现：

* `NewModel(const model_args & args)`：保存初始化参数，校验新模型必需的权重路径或配置，并初始化模型内部 runtime、权重、cache 或组件对象。
* `~NewModel()`：释放构造函数里创建的资源，保证模型对象销毁时不会泄漏 context、backend buffer 或外部 runtime 句柄。
* `type()`：返回稳定的模型类型字符串，例如 `"newmodel"`，方便日志和调试。
* `predict()`：校验 `observation` 里的图片、state、task，执行一次模型推理，并把内部结果转换成 `model_result`。
* `reset()`：如果模型有 cache、时序状态或复用的采样状态，在这里清理；无状态模型可以留空。
* `is_ready()`：返回初始化是否成功，供 factory 在创建后检查。
* `make_newmodel()`：创建 `NewModel` 实例，处理分配失败或初始化失败，并通过 `error` 返回可读错误信息。

#### GGUF/ggml 初始化复用

通常在初始化中会需要有相关的ggml backend和gguf load的工序，这些工序通常来说都是高度相似的，因此我们使用公共函数用来直接调用。

具体而言，gguf loader需要针对不同的model去继承gguf loader的基类，主要需要重写bind tensor和parse_metadata部分，因为不同的gguf在这两部分通常都有其特性。

而ggml backend可以使用，其封装了backend初始化，scheduler的初始化，buft policy的初始化。

### 3. 扩展 `model_args` 和 `model_type`

在 `src/models/model.h` 中：

1. 给 `model_type` 增加新枚举，例如 `newmodel`。
2. 在 `model_args` 中增加新模型需要的参数，例如 `std::string newmodel_path;`。

如果参数能复用现有字段，例如 `llm_path`、`mmproj_path`、`state_path`，可以直接复用；如果语义不同，建议新增清晰命名的字段，避免后续维护时误解。上面的初始化骨架用 `args_.newmodel_path` 表示第 3 步新增的新模型权重路径字段。

### 4. 接入 factory

在 `src/models/model_factory.cpp` 中 include 新的model子类，并添加分发：

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

这样 `model-cli` 和 `model-server` 都可以通过统一入口创建新模型。

### 5. 接入 `model-cli`

在 `src/model-cli.cpp` 中至少需要修改：

* `parse_model_type()`：识别新的 `--model-type newmodel`。
* `print_usage()`：补充新模型的参数说明。
* 参数解析循环：解析新模型需要的 CLI 参数，并写入 `model_args`。

如果新模型对 image name、state 维度或 task 文本有特殊要求，尽量在子类的 `predict()` 或初始化校验里返回明确错误信息，方便 CLI 和 server 复用。

### 6. 修改 CMake

在 `CMakeLists.txt` 中把新模型文件加入合适的 target。最小接入只需要让 `robotcpp` 编译并链接 `newmodel_model.*` 以及它直接依赖的内部源文件：

```cmake
add_library(robotcpp STATIC
    src/models/model.h
    src/models/model_factory.cpp
    src/models/newmodel/newmodel_model.cpp
    src/models/newmodel/newmodel_model.h
    ...
)
```

### 7. 验证

#### 新模型最小验证模板

接入新模型后，也建议先用 `model-cli` 跑一次最小推理：

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

如果 `model-server` 也要使用该模型，还需要确认 server 启动参数能把新模型所需字段传到 `model_args`。

## 新模型接入检查清单

* `tools/hf2gguf/<model_name>/` 已能把原始 checkpoint 或 `.pt` 转成 C++ runtime 使用的 GGUF。
* `tools/quant/config/<model_name>_origin.yaml` 已覆盖新模型的所有 GGUF component 和 tensor group。
* `src/models/<model_name>/` 下有清晰的 `Model` ==子类==；只有在确实需要时才拆出内部 runtime 或 engine。
* `model_type` 已增加新枚举。
* `model_args` 已包含新模型初始化所需参数。
* `model_factory.cpp` 已接入新模型。
* `model-cli.cpp` 已支持 `--model-type <model_name>` 和必要参数。
* `CMakeLists.txt` 已加入新模型源文件，并让 `robotcpp` 能链接到必要的内部组件。
