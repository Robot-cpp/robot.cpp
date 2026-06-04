# SmolVLA GGUF Converter使用说明

这里放 SmolVLA safetensor checkpoint 到 GGUF 的转换工具。

* [X] 当前只处理 SmolVLA f32 f16
* [ ] 支持其他精度，譬如bf16
* [ ] 抽象支持其他model

SmolVLA runtime 长期按四个 GGUF 文件加载：

- `smolvla-llm-<dtype>.gguf`
- `mmproj-smolvla-<dtype>.gguf`
- `state-proj-smolvla-<dtype>.gguf`
- `action-expert-smolvla-<dtype>.gguf`
