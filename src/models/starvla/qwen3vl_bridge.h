#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp::starvla {

enum class QwenVLArchitecture {
    unknown,
    qwen2_5_vl,
    qwen3_vl,
};

enum class QwenVLHiddenStateSourceKind {
    decoder_output,
    deepstack_output,
    final_norm,
};

struct QwenVLHiddenStateSource {
    QwenVLHiddenStateSourceKind kind = QwenVLHiddenStateSourceKind::decoder_output;
    int layer                        = -1;
};

// Resolve the paired llama.cpp text and mtmd projector profiles. StarVLA
// supports Qwen2.5-VL and Qwen3-VL only; mismatched text/mmproj files fail
// before either component is evaluated.
bool qwen_vl_resolve_architecture(const std::string & text_architecture, const std::string & projector_type,
                                  QwenVLArchitecture & architecture, std::string & error);

const char * qwen_vl_architecture_name(QwenVLArchitecture architecture);

// llama.cpp names the final normalized decoder state `result_norm`, then
// renames the same tensor when embedding output with pooling type NONE is
// enabled. Both names identify the Qwen2.5 hidden_states[-1] boundary.
bool qwen_vl_is_final_norm_tensor_name(const char * name) noexcept;

// Map one Transformers 4.57 hidden_states tuple index to the corresponding
// llama.cpp graph output. Index zero (the embedding input) is intentionally not
// exposed. Qwen2.5-VL has no DeepStack and its final tuple item is result_norm;
// Qwen3-VL additionally maps the DeepStack outputs exposed by llama.cpp.
bool qwen_vl_hidden_state_source(QwenVLArchitecture architecture, int decoder_layer_count, int deepstack_layer_count,
                                 int32_t hidden_tuple_index, QwenVLHiddenStateSource & source, std::string & error);

struct Qwen3VLImageView {
    const uint8_t * data = nullptr;
    int width            = 0;
    int height           = 0;
    int channels         = 0;
    int stride_bytes     = 0;
};

struct Qwen3VLBridgeConfig {
    std::string text_path;
    std::string mmproj_path;
    std::string bundle_uuid;
    int hidden_size          = 0;
    int input_embedding_size = 0;
    int vocab_size           = 0;
    std::string action_token;
    int32_t action_token_id      = -1;
    int expected_image_count     = 0;
    int image_min_tokens         = 0;
    int image_max_tokens         = 0;
    int image_spatial_merge_size = 0;
    int n_ctx                    = 2048;
    int n_batch                  = 2048;
    int n_threads                = 0;
    int verbosity                = 0;
    // OFT uses flash attention; other variants require intermediate outputs
    // that are only available on the non-flash path.
    bool flash_text_attention = false;
    // Round each F32 decoder residual output, plus DeepStack outputs when the
    // detected architecture has them, through BF16 RNE before it feeds the next
    // layer. Intra-layer computation keeps llama.cpp's backend-default profile.
    bool bf16_residual_layer_boundaries = false;
    // Repeated text decode can rebuild graphs with transient node keys. Disable
    // native graph capture/cache for the text context so those keys cannot
    // accumulate backend graph instances. Direct graph computation continues.
    bool disable_text_backend_native_graphs = false;
    // The vision encoder rebuilds its graph for every image. Disable native
    // graph capture/cache when its transient graph keys are not stable.
    bool disable_vision_backend_native_graphs = false;
};

struct QwenVLGenerationConfig {
    size_t max_length = 0;
    std::vector<int32_t> eos_token_ids;
    int top_k                = 0;
    float repetition_penalty = 0.0f;
};

struct QwenVLGenerationResult {
    size_t prompt_token_count = 0;
    std::vector<int32_t> full_sequence;
    std::vector<int32_t> continuation;
};

// Implements the deterministic token choice used by the official FAST
// generation profile: Hugging Face repetition penalty over the full sequence,
// followed by top_k=1. Exposed so the generation contract can be tested
// without loading a multi-gigabyte Qwen checkpoint.
bool qwen_vl_select_repetition_penalized_top1(const float * logits, size_t vocab_size,
                                              const std::vector<int32_t> & full_sequence, float repetition_penalty,
                                              int32_t & token, std::string & error);

class Qwen3VLBridge {
  public:
    ~Qwen3VLBridge();

    Qwen3VLBridge(const Qwen3VLBridge &)             = delete;
    Qwen3VLBridge & operator=(const Qwen3VLBridge &) = delete;

    static std::unique_ptr<Qwen3VLBridge> load(const Qwen3VLBridgeConfig & config, std::string & error);

    bool extract_token_embeddings(const std::vector<Qwen3VLImageView> & images, const std::string & instruction,
                                  int32_t token_id, size_t token_count, std::vector<float> & embeddings,
                                  std::string & error);

    // Full conditioning sequence. Qwen3 uses the outer recorder's raw final
    // decoder output (`l_out-(N-1)`); Qwen2.5 uses `result_norm`, matching its
    // Transformers hidden_states[-1]. Values are widened from BF16.
    bool extract_full_hidden_states(const std::vector<Qwen3VLImageView> & images, const std::string & instruction,
                                    std::vector<float> & hidden_states, std::vector<uint8_t> & attention_mask,
                                    std::string & error);

    // hidden_tuple_indices use the pinned Transformers 4.57 convention. Index
    // zero is the embedding input and is not exposed. For Qwen3, in-place
    // DeepStack aliases make the first D entries `deepstack_out`; remaining
    // entries, including N, are raw `l_out`. For Qwen2.5, indices 1..N-1 are
    // raw `l_out` and index N is `result_norm`. The result is layer-major
    // [requested states, tokens, hidden size].
    bool extract_layer_hidden_states(const std::vector<Qwen3VLImageView> & images, const std::string & instruction,
                                     const std::vector<int32_t> & hidden_tuple_indices,
                                     std::vector<float> & hidden_states, std::vector<uint8_t> & attention_mask,
                                     std::string & error);

    // Runs a full multimodal prefill followed by incremental KV-cached text
    // decoding. The returned sequence includes the prompt, matching
    // Transformers generate(return_dict_in_generate=false).
    bool generate_autoregressive(const std::vector<Qwen3VLImageView> & images, const std::string & instruction,
                                 const QwenVLGenerationConfig & generation, QwenVLGenerationResult & result,
                                 std::string & error);

    void reset();
    const char * backend_name() const;
    QwenVLArchitecture architecture() const;

  private:
    struct Impl;

    explicit Qwen3VLBridge(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace robotcpp::starvla
