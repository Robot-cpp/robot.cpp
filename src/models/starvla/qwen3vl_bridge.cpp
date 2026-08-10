#include "models/starvla/qwen3vl_bridge.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-model.h"
#include "mtmd.h"
#include "models/starvla/oft_prompt.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace robotcpp::starvla {

bool qwen_vl_resolve_architecture(const std::string & text_architecture,
                                  const std::string & projector_type,
                                  QwenVLArchitecture & architecture,
                                  std::string & error) {
    architecture = QwenVLArchitecture::unknown;
    error.clear();
    if (text_architecture == "qwen2vl" &&
        projector_type == "qwen2.5vl_merger") {
        architecture = QwenVLArchitecture::qwen2_5_vl;
        return true;
    }
    if (text_architecture == "qwen3vl" &&
        projector_type == "qwen3vl_merger") {
        architecture = QwenVLArchitecture::qwen3_vl;
        return true;
    }
    if (text_architecture != "qwen2vl" &&
        text_architecture != "qwen3vl") {
        error = "unsupported Qwen-VL text architecture: " + text_architecture;
    } else if (projector_type != "qwen2.5vl_merger" &&
               projector_type != "qwen3vl_merger") {
        error = "unsupported Qwen-VL projector type: " + projector_type;
    } else {
        error = "Qwen-VL text and mmproj architectures do not match";
    }
    return false;
}

const char * qwen_vl_architecture_name(QwenVLArchitecture architecture) {
    switch (architecture) {
    case QwenVLArchitecture::qwen2_5_vl:
        return "qwen2.5-vl";
    case QwenVLArchitecture::qwen3_vl:
        return "qwen3-vl";
    case QwenVLArchitecture::unknown:
        break;
    }
    return "unknown";
}

bool qwen_vl_is_final_norm_tensor_name(const char * name) noexcept {
    return name != nullptr &&
           (std::strcmp(name, "result_norm") == 0 ||
            std::strcmp(name, "result_embd_pooled") == 0);
}

bool qwen_vl_hidden_state_source(QwenVLArchitecture architecture,
                                 int decoder_layer_count,
                                 int deepstack_layer_count,
                                 int32_t hidden_tuple_index,
                                 QwenVLHiddenStateSource & source,
                                 std::string & error) {
    source = QwenVLHiddenStateSource{};
    error.clear();
    if (decoder_layer_count <= 0 || hidden_tuple_index <= 0 ||
        hidden_tuple_index > decoder_layer_count) {
        error = "Qwen-VL hidden-state tuple index is out of range";
        return false;
    }
    if (architecture == QwenVLArchitecture::qwen2_5_vl) {
        if (deepstack_layer_count != 0) {
            error = "Qwen2.5-VL hidden-state profile cannot contain DeepStack";
            return false;
        }
        if (hidden_tuple_index == decoder_layer_count) {
            source.kind = QwenVLHiddenStateSourceKind::final_norm;
            source.layer = -1;
        } else {
            source.kind = QwenVLHiddenStateSourceKind::decoder_output;
            source.layer = hidden_tuple_index - 1;
        }
        return true;
    }
    if (architecture == QwenVLArchitecture::qwen3_vl) {
        if (deepstack_layer_count <= 0 ||
            deepstack_layer_count > decoder_layer_count) {
            error = "Qwen3-VL DeepStack layer count is incompatible with the model";
            return false;
        }
        source.kind = hidden_tuple_index <= deepstack_layer_count
                          ? QwenVLHiddenStateSourceKind::deepstack_output
                          : QwenVLHiddenStateSourceKind::decoder_output;
        source.layer = hidden_tuple_index - 1;
        return true;
    }
    error = "Qwen-VL hidden-state architecture is unknown";
    return false;
}

bool qwen_vl_select_repetition_penalized_top1(
    const float * logits, size_t vocab_size,
    const std::vector<int32_t> & full_sequence, float repetition_penalty,
    int32_t & token, std::string & error) {
    token = -1;
    error.clear();
    if (logits == nullptr || vocab_size == 0 ||
        vocab_size > static_cast<size_t>(INT32_MAX) ||
        !std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f) {
        error = "Qwen-VL generation selector received an invalid contract";
        return false;
    }

    std::vector<uint8_t> repeated(vocab_size, uint8_t{0});
    for (int32_t value : full_sequence) {
        if (value < 0 || static_cast<size_t>(value) >= vocab_size) {
            error = "Qwen-VL generated sequence contains an out-of-vocabulary token";
            return false;
        }
        repeated[static_cast<size_t>(value)] = 1;
    }

    float best = -std::numeric_limits<float>::infinity();
    int32_t best_token = -1;
    for (size_t index = 0; index < vocab_size; ++index) {
        float score = logits[index];
        if (std::isnan(score)) {
            error = "Qwen-VL generation logits contain NaN";
            return false;
        }
        if (repeated[index] != 0) {
            score = score < 0.0f ? score * repetition_penalty
                                 : score / repetition_penalty;
        }
        // torch.argmax returns the first index on ties.
        if (best_token < 0 || score > best) {
            best = score;
            best_token = static_cast<int32_t>(index);
        }
    }
    if (best_token < 0) {
        error = "Qwen-VL generation selector did not produce a token";
        return false;
    }
    token = best_token;
    return true;
}

namespace {

void quiet_mtmd_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

struct PreparedMultimodalBatch {
    size_t token_count = 0;
    llama_pos position_count = 0;
    std::vector<float> embeddings;
    std::vector<llama_pos> positions;
    std::vector<int32_t> sequence_counts;
    std::vector<llama_seq_id> sequence_values;
    std::vector<llama_seq_id *> sequences;
    std::vector<int8_t> outputs;
    std::vector<llama_token> token_ids;

    llama_batch view() {
        return {
            static_cast<int32_t>(token_count),
            nullptr,
            embeddings.data(),
            positions.data(),
            sequence_counts.data(),
            sequences.data(),
            outputs.data(),
        };
    }
};

struct BackendPlacement {
    bool accelerator_compute = false;
    bool cpu_compute = false;
};

struct LayerCapture {
    BackendPlacement * placement = nullptr;
    bool enabled = false;
    bool bf16_residual_layer_boundaries = false;
    size_t expected_deepstack_layer_count = 0;
    size_t token_count = 0;
    size_t hidden_size = 0;
    std::vector<int> layer_to_slot;
    std::vector<int> deepstack_to_slot;
    int result_norm_slot = -1;
    std::vector<float> values;
    std::vector<uint8_t> seen;
    std::vector<uint8_t> rounded_layers;
    std::vector<uint8_t> rounded_deepstack_layers;
    std::string error;

    void disable() {
        enabled = false;
        token_count = 0;
        hidden_size = 0;
        layer_to_slot.clear();
        deepstack_to_slot.clear();
        result_norm_slot = -1;
        values.clear();
        seen.clear();
        rounded_layers.clear();
        rounded_deepstack_layers.clear();
        error.clear();
    }
};

void begin_layer_boundary_tracking(LayerCapture & capture, size_t layer_count) {
    if (!capture.bf16_residual_layer_boundaries) {
        capture.rounded_layers.clear();
        capture.rounded_deepstack_layers.clear();
        return;
    }
    capture.rounded_layers.assign(layer_count, uint8_t{0});
    capture.rounded_deepstack_layers.assign(
        capture.expected_deepstack_layer_count, uint8_t{0});
}

bool validate_layer_boundary_tracking(const LayerCapture & capture, std::string & error) {
    if (!capture.bf16_residual_layer_boundaries) {
        return true;
    }
    if (capture.rounded_layers.size() != capture.layer_to_slot.size() ||
        capture.rounded_deepstack_layers.size() !=
            capture.expected_deepstack_layer_count ||
        std::any_of(capture.rounded_layers.begin(), capture.rounded_layers.end(),
                    [](uint8_t seen) { return seen != 1; }) ||
        std::any_of(capture.rounded_deepstack_layers.begin(),
                    capture.rounded_deepstack_layers.end(),
                    [](uint8_t seen) { return seen != 1; })) {
        error =
            "Qwen3-VL BF16 residual-boundary roundtrip did not observe "
            "every expected l_out/deepstack_out exactly once";
        return false;
    }
    return true;
}

bool observe_backend_placement(ggml_tensor * tensor, bool ask, void * user_data) {
    if (!ask || tensor == nullptr || tensor->op == GGML_OP_NONE || tensor->buffer == nullptr ||
        user_data == nullptr) {
        return false;
    }
    auto * placement = static_cast<BackendPlacement *>(user_data);
    ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(tensor->buffer);
    ggml_backend_dev_t device =
        buffer_type == nullptr ? nullptr : ggml_backend_buft_get_device(buffer_type);
    if (device == nullptr) {
        return false;
    }
    switch (ggml_backend_dev_type(device)) {
    case GGML_BACKEND_DEVICE_TYPE_GPU:
    case GGML_BACKEND_DEVICE_TYPE_IGPU:
    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
        placement->accelerator_compute = true;
        break;
    case GGML_BACKEND_DEVICE_TYPE_CPU:
        placement->cpu_compute = true;
        break;
    case GGML_BACKEND_DEVICE_TYPE_META:
        break;
    }
    return false;
}

int indexed_output_index(const char * name, const char * prefix) {
    const size_t prefix_size = std::strlen(prefix);
    if (name == nullptr || std::strncmp(name, prefix, prefix_size) != 0) {
        return -1;
    }
    const char * number = name + prefix_size;
    if (*number == '\0') {
        return -1;
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(number, &end, 10);
    if (errno != 0 || end == number || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return -1;
    }
    return static_cast<int>(parsed);
}

bool observe_text_and_capture_layers(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<LayerCapture *>(user_data);
    if (capture == nullptr) {
        return false;
    }
    observe_backend_placement(tensor, ask, capture->placement);
    if (!capture->enabled || tensor == nullptr) {
        return false;
    }

    int slot = -1;
    const int deepstack_layer = indexed_output_index(tensor->name, "deepstack_out-");
    const int layer = indexed_output_index(tensor->name, "l_out-");
    const bool is_result_norm =
        qwen_vl_is_final_norm_tensor_name(tensor->name);
    const bool valid_deepstack_layer =
        deepstack_layer >= 0 &&
        static_cast<size_t>(deepstack_layer) < capture->deepstack_to_slot.size();
    const bool valid_layer =
        layer >= 0 && static_cast<size_t>(layer) < capture->layer_to_slot.size();
    if (valid_deepstack_layer) {
        slot = capture->deepstack_to_slot[static_cast<size_t>(deepstack_layer)];
    } else if (valid_layer) {
        slot = capture->layer_to_slot[static_cast<size_t>(layer)];
    } else if (is_result_norm) {
        slot = capture->result_norm_slot;
    }
    const bool round_layer =
        capture->bf16_residual_layer_boundaries && valid_layer;
    const bool round_deepstack =
        capture->bf16_residual_layer_boundaries && valid_deepstack_layer &&
        static_cast<size_t>(deepstack_layer) <
            capture->expected_deepstack_layer_count;
    if (slot < 0 && !round_layer && !round_deepstack) {
        return false;
    }
    if (ask) {
        return true;
    }

    try {
        if (slot >= 0) {
            if (static_cast<size_t>(slot) >= capture->seen.size()) {
                capture->error = "Qwen3-VL layer-capture slot is out of range";
                return false;
            }
            if (capture->seen[static_cast<size_t>(slot)] != 0) {
                capture->error =
                    "Qwen3-VL emitted a requested hidden-state output more than once";
                return false;
            }
        }
        if (round_layer &&
            (static_cast<size_t>(layer) >= capture->rounded_layers.size() ||
             capture->rounded_layers[static_cast<size_t>(layer)] != 0)) {
            capture->error =
                "Qwen3-VL l_out BF16 roundtrip index is invalid or repeated";
            return false;
        }
        if (round_deepstack &&
            (static_cast<size_t>(deepstack_layer) >=
                 capture->rounded_deepstack_layers.size() ||
             capture->rounded_deepstack_layers[static_cast<size_t>(deepstack_layer)] != 0)) {
            capture->error =
                "Qwen3-VL deepstack_out BF16 roundtrip index is invalid or repeated";
            return false;
        }
        if (!ggml_is_contiguous(tensor) || tensor->ne[0] != static_cast<int64_t>(capture->hidden_size) ||
            tensor->ne[1] != static_cast<int64_t>(capture->token_count) || tensor->ne[2] != 1 ||
            tensor->ne[3] != 1) {
            capture->error = "Qwen3-VL hidden-state output has an incompatible shape or layout";
            return false;
        }
        if (capture->hidden_size == 0 ||
            capture->token_count >
                std::numeric_limits<size_t>::max() / capture->hidden_size) {
            capture->error = "Qwen3-VL hidden-state capture size overflow";
            return false;
        }
        const size_t count = capture->token_count * capture->hidden_size;
        if (count > std::numeric_limits<size_t>::max() / sizeof(float) ||
            (slot >= 0 &&
             (count == 0 || static_cast<size_t>(slot) >= capture->values.size() / count))) {
            capture->error = "Qwen3-VL hidden-state capture byte range is invalid";
            return false;
        }
        std::vector<float> rounded(count);
        if (tensor->type == GGML_TYPE_F32) {
            std::vector<float> source(count);
            ggml_backend_tensor_get(tensor, source.data(), 0, count * sizeof(float));
            for (size_t index = 0; index < count; ++index) {
                rounded[index] =
                    ggml_bf16_to_fp32(ggml_fp32_to_bf16(source[index]));
            }
        } else if (tensor->type == GGML_TYPE_F16) {
            if (round_layer || round_deepstack) {
                capture->error =
                    "Qwen3-VL BF16 residual-boundary roundtrip requires F32 tensors";
                return false;
            }
            std::vector<ggml_fp16_t> source(count);
            ggml_backend_tensor_get(tensor, source.data(), 0,
                                    count * sizeof(ggml_fp16_t));
            for (size_t index = 0; index < count; ++index) {
                rounded[index] = ggml_bf16_to_fp32(
                    ggml_fp32_to_bf16(ggml_fp16_to_fp32(source[index])));
            }
        } else if (tensor->type == GGML_TYPE_BF16) {
            if (round_layer || round_deepstack) {
                capture->error =
                    "Qwen3-VL BF16 residual-boundary roundtrip requires F32 tensors";
                return false;
            }
            std::vector<ggml_bf16_t> source(count);
            ggml_backend_tensor_get(tensor, source.data(), 0,
                                    count * sizeof(ggml_bf16_t));
            for (size_t index = 0; index < count; ++index) {
                rounded[index] = ggml_bf16_to_fp32(source[index]);
            }
        } else {
            capture->error = std::string("unsupported Qwen3-VL hidden-state output type: ") +
                             ggml_type_name(tensor->type);
            return false;
        }
        if (round_layer || round_deepstack) {
            ggml_backend_tensor_set(tensor, rounded.data(), 0,
                                    count * sizeof(float));
            if (round_layer) {
                capture->rounded_layers[static_cast<size_t>(layer)] = 1;
            } else {
                capture->rounded_deepstack_layers[
                    static_cast<size_t>(deepstack_layer)] = 1;
            }
        }
        if (slot >= 0) {
            float * destination =
                capture->values.data() + static_cast<size_t>(slot) * count;
            std::copy(rounded.begin(), rounded.end(), destination);
            capture->seen[static_cast<size_t>(slot)] = 1;
        }
        return true;
    } catch (const std::exception & exception) {
        capture->error = std::string("failed to capture Qwen3-VL hidden-state output: ") +
                         exception.what();
        return false;
    } catch (...) {
        capture->error = "failed to capture Qwen3-VL hidden-state output";
        return false;
    }
}

int32_t decode_and_synchronize(llama_context * context, llama_batch batch) {
    const int32_t result = llama_decode(context, batch);
    // A cb_eval capture synchronizes only through its selected node. Drain the
    // remaining graph tail before disabling capture, clearing KV state, or
    // entering a downstream policy graph.
    llama_synchronize(context);
    return result;
}

std::string model_metadata(const llama_model * model, const char * key) {
    char value[256] = {};
    const int32_t length = llama_model_meta_val_str(model, key, value, sizeof(value));
    if (length < 0 || static_cast<size_t>(length) >= sizeof(value)) {
        throw std::runtime_error(std::string("missing or oversized Qwen GGUF metadata: ") + key);
    }
    return value;
}

std::string gguf_string_metadata(const std::string & path, const char * key) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr) {
        throw std::runtime_error("failed to read GGUF metadata: " + path);
    }
    const int64_t index = gguf_find_key(gguf, key);
    if (index < 0 || gguf_get_kv_type(gguf, index) != GGUF_TYPE_STRING) {
        gguf_free(gguf);
        throw std::runtime_error(std::string("missing GGUF string metadata ") + key + ": " + path);
    }
    const std::string value = gguf_get_val_str(gguf, index);
    gguf_free(gguf);
    return value;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
                                  bool parse_special) {
    const int32_t required = -llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                              nullptr, 0, false, parse_special);
    if (required <= 0) {
        return {};
    }
    std::vector<llama_token> tokens(static_cast<size_t>(required));
    const int32_t count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                         tokens.data(), required, false, parse_special);
    if (count != required) {
        return {};
    }
    return tokens;
}

std::string apply_chat_template(const llama_model * model,
                                QwenVLArchitecture architecture,
                                const std::string & content) {
    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template == nullptr || chat_template[0] == '\0') {
        throw std::runtime_error("Qwen text GGUF has no default chat template");
    }
    const llama_chat_message messages[] = {
        {"system", "You are a helpful assistant."},
        {"user", content.c_str()},
    };
    const size_t message_offset =
        architecture == QwenVLArchitecture::qwen2_5_vl ? 0U : 1U;
    const size_t message_count =
        architecture == QwenVLArchitecture::qwen2_5_vl ? 2U : 1U;
    const int32_t required =
        llama_chat_apply_template(chat_template, messages + message_offset,
                                  message_count, true, nullptr, 0);
    if (required < 0) {
        throw std::runtime_error("failed to size the Qwen chat-template output");
    }
    std::vector<char> buffer(static_cast<size_t>(required) + 1, '\0');
    if (buffer.size() > static_cast<size_t>(INT32_MAX)) {
        throw std::runtime_error("Qwen chat-template output is too large");
    }
    const int32_t written = llama_chat_apply_template(
        chat_template, messages + message_offset, message_count, true,
        buffer.data(), static_cast<int32_t>(buffer.size()));
    if (written != required) {
        throw std::runtime_error("failed to apply the Qwen chat template");
    }
    return std::string(buffer.data(), static_cast<size_t>(written));
}

struct PackedImageLayout {
    size_t row_bytes = 0;
    size_t stride_bytes = 0;
    size_t packed_bytes = 0;
};

bool validate_image(const Qwen3VLImageView & image, const Qwen3VLBridgeConfig & config,
                    PackedImageLayout & layout, std::string & error) {
    (void) config;
    layout = PackedImageLayout{};
    if (image.data == nullptr || image.channels != 3 || image.width <= 0 ||
        image.height <= 0) {
        error = "Qwen3-VL bridge requires a non-empty RGB image";
        return false;
    }
    const size_t width = static_cast<size_t>(image.width);
    const size_t height = static_cast<size_t>(image.height);
    if (width > std::numeric_limits<size_t>::max() / 3U) {
        error = "Qwen3-VL image row size overflow";
        return false;
    }
    layout.row_bytes = width * 3U;
    if (image.stride_bytes < 0 ||
        (image.stride_bytes > 0 &&
         static_cast<size_t>(image.stride_bytes) < layout.row_bytes)) {
        error = "Qwen3-VL image stride is smaller than a packed RGB row";
        return false;
    }
    layout.stride_bytes = image.stride_bytes > 0
                              ? static_cast<size_t>(image.stride_bytes)
                              : layout.row_bytes;
    if (height > std::numeric_limits<size_t>::max() / layout.row_bytes ||
        (height > 1U &&
         height - 1U >
             (std::numeric_limits<size_t>::max() - layout.row_bytes) /
                 layout.stride_bytes)) {
        error = "Qwen3-VL image buffer size overflow";
        return false;
    }
    layout.packed_bytes = height * layout.row_bytes;
    return true;
}

std::vector<uint8_t> pack_image(const Qwen3VLImageView & image,
                                const PackedImageLayout & layout) {
    std::vector<uint8_t> packed(layout.packed_bytes);
    for (int row = 0; row < image.height; ++row) {
        std::memcpy(packed.data() + static_cast<size_t>(row) * layout.row_bytes,
                    image.data + static_cast<size_t>(row) * layout.stride_bytes,
                    layout.row_bytes);
    }
    return packed;
}

void tokenize_multimodal_prompt(const Qwen3VLBridgeConfig & config,
                                QwenVLArchitecture architecture,
                                const llama_model * model, mtmd_context * vision,
                                const std::vector<Qwen3VLImageView> & images,
                                const std::string & instruction,
                                mtmd::input_chunks & chunks) {
    std::vector<std::vector<uint8_t>> packed_images;
    packed_images.reserve(images.size());
    mtmd::bitmaps bitmaps;
    for (const Qwen3VLImageView & image : images) {
        std::string validation_error;
        PackedImageLayout layout;
        if (!validate_image(image, config, layout, validation_error)) {
            throw std::runtime_error(validation_error);
        }
        packed_images.push_back(pack_image(image, layout));
        bitmaps.entries.emplace_back(static_cast<uint32_t>(image.width),
                                     static_cast<uint32_t>(image.height),
                                     packed_images.back().data());
        if (bitmaps.entries.back().ptr == nullptr) {
            throw std::runtime_error("failed to create a Qwen3-VL image bitmap");
        }
    }

    const std::string content =
        build_qwen_media_content(images.size(), instruction, mtmd_default_marker());
    const std::string formatted =
        apply_chat_template(model, architecture, content);
    mtmd_input_text input_text{};
    input_text.text = formatted.c_str();
    input_text.add_special = false;
    input_text.parse_special = true;
    chunks.ptr.reset(mtmd_input_chunks_init());
    if (chunks.ptr == nullptr) {
        throw std::runtime_error("failed to allocate Qwen3-VL multimodal input chunks");
    }
    std::vector<const mtmd_bitmap *> bitmap_ptrs = bitmaps.c_ptr();
    const int32_t tokenize_result = mtmd_tokenize(
        vision, chunks.ptr.get(), &input_text, bitmap_ptrs.data(), bitmap_ptrs.size());
    if (tokenize_result != 0) {
        throw std::runtime_error("failed to tokenize the Qwen3-VL multimodal prompt");
    }
}

const char * compiled_backend_name() {
#if defined(GGML_USE_CUDA)
    return "cuda";
#elif defined(GGML_USE_METAL)
    return "metal";
#else
    return "cpu";
#endif
}

} // namespace

struct Qwen3VLBridge::Impl {
    Qwen3VLBridgeConfig config;
    QwenVLArchitecture architecture = QwenVLArchitecture::unknown;
    size_t deepstack_layer_count = 0;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    mtmd_context * vision = nullptr;
    const llama_vocab * vocab = nullptr;
    bool backend_initialized = false;
    BackendPlacement text_placement;
    BackendPlacement vision_placement;
    LayerCapture layer_capture;
    mutable std::string backend_name = "unknown";

    void refresh_backend_name() const {
        const bool accelerator =
            text_placement.accelerator_compute && vision_placement.accelerator_compute;
        const bool cpu = text_placement.cpu_compute || vision_placement.cpu_compute;
        if (accelerator && !cpu) {
            backend_name = compiled_backend_name();
        } else if (!text_placement.accelerator_compute &&
                   !vision_placement.accelerator_compute &&
                   text_placement.cpu_compute && vision_placement.cpu_compute) {
            backend_name = "cpu";
        } else if (text_placement.accelerator_compute ||
                   vision_placement.accelerator_compute) {
            backend_name = "mixed";
        } else {
            backend_name = "unknown";
        }
    }

    ~Impl() {
        if (vision != nullptr) {
            mtmd_free(vision);
            vision = nullptr;
        }
        if (context != nullptr) {
            llama_free(context);
            context = nullptr;
        }
        if (model != nullptr) {
            llama_model_free(model);
            model = nullptr;
        }
        if (backend_initialized) {
            llama_backend_free();
            backend_initialized = false;
        }
    }
};

namespace {

void copy_token_embedding(const ggml_tensor * token_embeddings, llama_token token,
                          size_t hidden_size, float * destination) {
    if (token_embeddings == nullptr || destination == nullptr || token < 0 ||
        token_embeddings->ne[0] != static_cast<int64_t>(hidden_size) ||
        token >= token_embeddings->ne[1] || !ggml_is_contiguous(token_embeddings) ||
        token_embeddings->buffer == nullptr) {
        throw std::runtime_error("Qwen3-VL token embedding table is incompatible");
    }
    const size_t row_stride = token_embeddings->nb[1];
    if (row_stride == 0 ||
        static_cast<size_t>(token) > std::numeric_limits<size_t>::max() / row_stride) {
        throw std::runtime_error("Qwen3-VL token embedding row offset overflow");
    }
    const size_t row_offset = static_cast<size_t>(token) * row_stride;
    switch (token_embeddings->type) {
    case GGML_TYPE_F32:
        ggml_backend_tensor_get(token_embeddings, destination, row_offset,
                                hidden_size * sizeof(float));
        break;
    case GGML_TYPE_F16: {
        std::vector<ggml_fp16_t> row(hidden_size);
        ggml_backend_tensor_get(token_embeddings, row.data(), row_offset,
                                hidden_size * sizeof(ggml_fp16_t));
        for (size_t index = 0; index < hidden_size; ++index) {
            destination[index] = ggml_fp16_to_fp32(row[index]);
        }
        break;
    }
    case GGML_TYPE_BF16: {
        std::vector<ggml_bf16_t> row(hidden_size);
        ggml_backend_tensor_get(token_embeddings, row.data(), row_offset,
                                hidden_size * sizeof(ggml_bf16_t));
        for (size_t index = 0; index < hidden_size; ++index) {
            destination[index] = ggml_bf16_to_fp32(row[index]);
        }
        break;
    }
    default:
        throw std::runtime_error(std::string("unsupported Qwen3-VL token embedding type: ") +
                                 ggml_type_name(token_embeddings->type));
    }
}

PreparedMultimodalBatch prepare_multimodal_batch(const Qwen3VLBridgeConfig & config,
                                                  llama_model * model,
                                                  mtmd_context * vision,
                                                  const mtmd::input_chunks & chunks) {
    if (model == nullptr || vision == nullptr || !mtmd_decode_use_mrope(vision)) {
        throw std::runtime_error("Qwen3-VL single-batch decode requires M-RoPE components");
    }

    PreparedMultimodalBatch prepared;
    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const mtmd_input_chunk * chunk = chunks[chunk_index];
        const size_t chunk_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        if (chunk_tokens == 0 || chunk_tokens > std::numeric_limits<size_t>::max() -
                                                  prepared.token_count) {
            throw std::runtime_error("Qwen3-VL multimodal chunk has an invalid token count");
        }
        prepared.token_count += chunk_tokens;
    }
    if (prepared.token_count == 0 || prepared.token_count > static_cast<size_t>(INT32_MAX)) {
        throw std::runtime_error("Qwen3-VL multimodal prompt token count is invalid");
    }

    const size_t hidden_size = static_cast<size_t>(config.hidden_size);
    const size_t input_size = static_cast<size_t>(config.input_embedding_size);
    if (hidden_size == 0 || input_size != static_cast<size_t>(llama_model_n_embd_inp(model)) ||
        input_size < hidden_size ||
        prepared.token_count > std::numeric_limits<size_t>::max() / input_size) {
        throw std::runtime_error("Qwen3-VL input embedding dimensions are incompatible");
    }
    if (prepared.token_count > std::numeric_limits<size_t>::max() / 4U) {
        throw std::runtime_error("Qwen3-VL M-RoPE position buffer size overflow");
    }

    prepared.embeddings.assign(prepared.token_count * input_size, 0.0f);
    prepared.positions.resize(prepared.token_count * 4U);
    prepared.sequence_counts.assign(prepared.token_count, 1);
    prepared.sequence_values.assign(prepared.token_count, 0);
    prepared.sequences.resize(prepared.token_count);
    prepared.outputs.assign(prepared.token_count, int8_t{0});
    prepared.token_ids.assign(prepared.token_count, static_cast<llama_token>(-1));
    for (size_t index = 0; index < prepared.token_count; ++index) {
        prepared.sequences[index] = &prepared.sequence_values[index];
    }

    const ggml_tensor * token_embeddings = model->get_tensor("token_embd.weight");
    size_t token_offset = 0;
    llama_pos position_offset = 0;
    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const mtmd_input_chunk * chunk = chunks[chunk_index];
        const size_t chunk_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        const llama_pos chunk_positions = mtmd_input_chunk_get_n_pos(chunk);
        if (chunk_positions <= 0 ||
            position_offset > std::numeric_limits<llama_pos>::max() - chunk_positions) {
            throw std::runtime_error("Qwen3-VL multimodal positions overflow");
        }

        const mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t text_token_count = 0;
            const llama_token * tokens =
                mtmd_input_chunk_get_tokens_text(chunk, &text_token_count);
            if (tokens == nullptr || text_token_count != chunk_tokens ||
                chunk_positions != static_cast<llama_pos>(chunk_tokens)) {
                throw std::runtime_error("Qwen3-VL text chunk contract is incompatible");
            }
            for (size_t local_index = 0; local_index < chunk_tokens; ++local_index) {
                const size_t global_index = token_offset + local_index;
                copy_token_embedding(token_embeddings, tokens[local_index], hidden_size,
                                     prepared.embeddings.data() + global_index * input_size);
                prepared.token_ids[global_index] = tokens[local_index];
                const llama_pos position =
                    position_offset + static_cast<llama_pos>(local_index);
                for (size_t axis = 0; axis < 3U; ++axis) {
                    prepared.positions[axis * prepared.token_count + global_index] = position;
                }
                prepared.positions[prepared.token_count * 3U + global_index] = 0;
            }
        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const mtmd_image_tokens * image_tokens =
                mtmd_input_chunk_get_tokens_image(chunk);
            if (image_tokens == nullptr ||
                mtmd_image_tokens_get_n_tokens(image_tokens) != chunk_tokens) {
                throw std::runtime_error("Qwen3-VL image chunk contract is incompatible");
            }
            if (mtmd_encode_chunk(vision, chunk) != 0) {
                throw std::runtime_error("failed to encode a Qwen3-VL image chunk");
            }
            const float * image_embeddings = mtmd_get_output_embd(vision);
            if (image_embeddings == nullptr) {
                throw std::runtime_error("Qwen3-VL image encoder returned no embeddings");
            }
            const size_t image_element_count = chunk_tokens * input_size;
            float * destination =
                prepared.embeddings.data() + token_offset * input_size;
            for (size_t element = 0; element < image_element_count; ++element) {
                destination[element] = ggml_bf16_to_fp32(
                    ggml_fp32_to_bf16(image_embeddings[element]));
            }
            for (size_t local_index = 0; local_index < chunk_tokens; ++local_index) {
                const size_t global_index = token_offset + local_index;
                const mtmd_decoder_pos position = mtmd_image_tokens_get_decoder_pos(
                    image_tokens, position_offset, local_index);
                prepared.positions[global_index] = static_cast<llama_pos>(position.t);
                prepared.positions[prepared.token_count + global_index] =
                    static_cast<llama_pos>(position.y);
                prepared.positions[prepared.token_count * 2U + global_index] =
                    static_cast<llama_pos>(position.x);
                prepared.positions[prepared.token_count * 3U + global_index] =
                    static_cast<llama_pos>(position.z);
            }
        } else {
            throw std::runtime_error("Qwen3-VL prompt contains an unsupported media chunk");
        }
        token_offset += chunk_tokens;
        position_offset += chunk_positions;
    }
    if (token_offset != prepared.token_count) {
        throw std::runtime_error("Qwen3-VL prepared batch token count mismatch");
    }
    prepared.position_count = position_offset;
    return prepared;
}

void export_prepared_inputs(const Qwen3VLBridgeConfig & config,
                            const llama_vocab * vocab,
                            const mtmd::input_chunks & chunks,
                            const PreparedMultimodalBatch & prepared,
                            std::vector<int64_t> & input_ids,
                            std::vector<uint8_t> & attention_mask,
                            std::vector<int64_t> & image_grid_thw) {
    const std::vector<llama_token> image_pad_tokens =
        tokenize(vocab, "<|image_pad|>", true);
    if (image_pad_tokens.size() != 1) {
        throw std::runtime_error(
            "Qwen3-VL vocabulary does not expose a unique <|image_pad|> token");
    }
    if (config.image_spatial_merge_size <= 0) {
        throw std::runtime_error("Qwen3-VL image spatial merge size is invalid");
    }

    input_ids.reserve(prepared.token_ids.size());
    for (llama_token token : prepared.token_ids) {
        input_ids.push_back(token < 0 ? static_cast<int64_t>(image_pad_tokens.front())
                                      : static_cast<int64_t>(token));
    }
    attention_mask.assign(prepared.token_count, uint8_t{1});

    image_grid_thw.reserve(static_cast<size_t>(config.expected_image_count) * 3U);
    size_t observed_images = 0;
    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const mtmd_input_chunk * chunk = chunks[chunk_index];
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            continue;
        }
        const mtmd_image_tokens * image_tokens =
            mtmd_input_chunk_get_tokens_image(chunk);
        if (image_tokens == nullptr) {
            throw std::runtime_error("Qwen3-VL image chunk has no token grid");
        }
        const size_t image_token_count =
            mtmd_image_tokens_get_n_tokens(image_tokens);
        uint32_t max_x = 0;
        uint32_t max_y = 0;
        for (size_t token = 0; token < image_token_count; ++token) {
            const mtmd_decoder_pos position =
                mtmd_image_tokens_get_decoder_pos(image_tokens, 0, token);
            if (position.t != 0 || position.z != 0) {
                throw std::runtime_error(
                    "Qwen3-VL image token grid does not use the expected M-RoPE layout");
            }
            max_x = std::max(max_x, position.x);
            max_y = std::max(max_y, position.y);
        }
        const size_t merged_width = static_cast<size_t>(max_x) + 1U;
        const size_t merged_height = static_cast<size_t>(max_y) + 1U;
        const size_t merge = static_cast<size_t>(config.image_spatial_merge_size);
        if (merged_width == 0 || merged_height == 0 ||
            merged_width > static_cast<size_t>(INT64_MAX) / merge ||
            merged_height > static_cast<size_t>(INT64_MAX) / merge ||
            merged_width > std::numeric_limits<size_t>::max() / merged_height ||
            merged_width * merged_height != image_token_count) {
            throw std::runtime_error("Qwen3-VL image token grid is incompatible");
        }
        image_grid_thw.push_back(1);
        image_grid_thw.push_back(static_cast<int64_t>(merged_height * merge));
        image_grid_thw.push_back(static_cast<int64_t>(merged_width * merge));
        ++observed_images;
    }
    if (observed_images != static_cast<size_t>(config.expected_image_count)) {
        throw std::runtime_error("Qwen3-VL image grid count does not match the policy");
    }
}

} // namespace

Qwen3VLBridge::Qwen3VLBridge(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Qwen3VLBridge::~Qwen3VLBridge() = default;

std::unique_ptr<Qwen3VLBridge> Qwen3VLBridge::load(const Qwen3VLBridgeConfig & config,
                                                   std::string & error) {
    error.clear();
    const bool action_config_valid =
        config.action_token.empty() ? config.action_token_id == -1 : config.action_token_id >= 0;
    if (config.text_path.empty() || config.mmproj_path.empty() || config.bundle_uuid.empty() ||
        config.hidden_size <= 0 || config.input_embedding_size <= 0 || config.vocab_size <= 0 ||
        !action_config_valid ||
        config.expected_image_count <= 0 || config.image_min_tokens <= 0 ||
        config.image_max_tokens < config.image_min_tokens ||
        config.image_spatial_merge_size <= 0 || config.n_ctx <= 0 ||
        config.n_batch <= 0) {
        error = "Qwen3-VL bridge configuration is incomplete";
        return nullptr;
    }

    std::unique_ptr<Impl> impl(new Impl());
    impl->config = config;
    try {
        const std::string mmproj_uuid =
            gguf_string_metadata(config.mmproj_path, "general.source.uuid");
        if (mmproj_uuid != config.bundle_uuid) {
            throw std::runtime_error("Qwen3-VL mmproj bundle UUID does not match the policy");
        }
        const std::string projector_type =
            gguf_string_metadata(config.mmproj_path, "clip.projector_type");

        llama_backend_init();
        impl->backend_initialized = true;
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        impl->model = llama_model_load_from_file(config.text_path.c_str(), model_params);
        if (impl->model == nullptr) {
            throw std::runtime_error("failed to load Qwen3-VL text GGUF: " + config.text_path);
        }
        if (model_metadata(impl->model, "general.source.uuid") != config.bundle_uuid) {
            throw std::runtime_error("Qwen3-VL text bundle UUID does not match the policy");
        }
        std::string profile_error;
        if (!qwen_vl_resolve_architecture(
                model_metadata(impl->model, "general.architecture"),
                projector_type, impl->architecture, profile_error)) {
            throw std::runtime_error(profile_error);
        }
        if (llama_model_n_embd_out(impl->model) != config.hidden_size ||
            llama_model_n_embd_inp(impl->model) != config.input_embedding_size) {
            throw std::runtime_error("Qwen3-VL text embedding dimensions do not match the policy");
        }
        if (config.input_embedding_size % config.hidden_size != 0) {
            throw std::runtime_error(
                "Qwen-VL input embedding width is not an integral hidden-state layout");
        }
        const int deepstack_layer_count =
            config.input_embedding_size / config.hidden_size - 1;
        if ((impl->architecture == QwenVLArchitecture::qwen2_5_vl &&
             deepstack_layer_count != 0) ||
            (impl->architecture == QwenVLArchitecture::qwen3_vl &&
             (deepstack_layer_count <= 0 ||
              deepstack_layer_count > llama_model_n_layer(impl->model)))) {
            throw std::runtime_error(
                "Qwen-VL input embedding layout does not match the detected architecture");
        }
        impl->deepstack_layer_count =
            static_cast<size_t>(deepstack_layer_count);
        for (int layer = 0; layer < llama_model_n_layer(impl->model); ++layer) {
            ggml_backend_dev_t device = impl->model->dev_layer(layer);
            if (device == nullptr) {
                throw std::runtime_error("Qwen3-VL text layer has no assigned backend device");
            }
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
            if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                impl->text_placement.cpu_compute = true;
            } else if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                       type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
                       type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                impl->text_placement.accelerator_compute = true;
            }
        }
        if (ggml_backend_dev_t output_device = impl->model->dev_output()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(output_device);
            if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                impl->text_placement.cpu_compute = true;
            } else if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                       type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
                       type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                impl->text_placement.accelerator_compute = true;
            }
        }

        impl->vocab = llama_model_get_vocab(impl->model);
        if (impl->vocab == nullptr) {
            throw std::runtime_error("Qwen3-VL text GGUF has no vocabulary");
        }
        if (llama_vocab_n_tokens(impl->vocab) != config.vocab_size) {
            throw std::runtime_error("Qwen3-VL text vocabulary size does not match the policy");
        }
        if (!config.action_token.empty()) {
            const std::vector<llama_token> action_tokens =
                tokenize(impl->vocab, config.action_token, true);
            if (action_tokens.size() != 1 || action_tokens.front() != config.action_token_id) {
                throw std::runtime_error(
                    "Qwen3-VL action token mapping does not match the policy metadata");
            }
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(config.n_ctx);
        context_params.n_batch = static_cast<uint32_t>(config.n_batch);
        // Layer capture expects one complete l_out/deepstack_out tensor per decode call.
        context_params.n_ubatch = static_cast<uint32_t>(config.n_batch);
        context_params.n_threads = config.n_threads;
        context_params.n_threads_batch = config.n_threads;
        context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        context_params.embeddings = false;
        // Match the official Qwen3-VL BF16 inference cache instead of llama's F16 default.
        context_params.type_k = GGML_TYPE_BF16;
        context_params.type_v = GGML_TYPE_BF16;
        context_params.flash_attn_type = config.flash_text_attention
            ? LLAMA_FLASH_ATTN_TYPE_ENABLED
            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        impl->layer_capture.placement = &impl->text_placement;
        impl->layer_capture.bf16_residual_layer_boundaries =
            config.bf16_residual_layer_boundaries;
        if (config.bf16_residual_layer_boundaries) {
            impl->layer_capture.expected_deepstack_layer_count =
                impl->deepstack_layer_count;
        }
        context_params.cb_eval = observe_text_and_capture_layers;
        context_params.cb_eval_user_data = &impl->layer_capture;
        impl->context = llama_init_from_model(impl->model, context_params);
        if (impl->context == nullptr) {
            throw std::runtime_error("failed to create Qwen3-VL text context");
        }
        if (config.disable_text_backend_native_graphs) {
            llama_set_backend_native_graphs_enabled(impl->context, false);
        }
        if (llama_n_batch(impl->context) != static_cast<uint32_t>(config.n_batch) ||
            llama_n_ubatch(impl->context) != static_cast<uint32_t>(config.n_batch)) {
            throw std::runtime_error(
                "Qwen3-VL text context did not preserve the requested batch/ubatch contract");
        }

        mtmd_context_params vision_params = mtmd_context_params_default();
        vision_params.use_gpu = true;
        vision_params.print_timings = config.verbosity >= 1;
        vision_params.n_threads = config.n_threads;
        vision_params.image_min_tokens = config.image_min_tokens;
        vision_params.image_max_tokens = config.image_max_tokens;
        vision_params.cb_eval = observe_backend_placement;
        vision_params.cb_eval_user_data = &impl->vision_placement;
        mtmd_log_set(config.verbosity >= 1 ? nullptr : quiet_mtmd_log_callback, nullptr);
        impl->vision = mtmd_init_from_file(config.mmproj_path.c_str(), impl->model, vision_params);
        if (impl->vision == nullptr) {
            throw std::runtime_error("failed to load Qwen3-VL mmproj GGUF: " + config.mmproj_path);
        }
        if (config.disable_vision_backend_native_graphs) {
            mtmd_set_backend_native_graphs_enabled(impl->vision, false);
        }
        impl->refresh_backend_name();

        if (config.verbosity >= 1) {
            std::fprintf(stderr,
                         "%s: architecture=%s backend=%s hidden=%d input_embd=%d "
                         "deepstack=%zu images=%d image_tokens=%d..%d "
                         "n_ctx=%u n_batch=%u n_ubatch=%u kv=bf16 "
                         "text_native_graph_disable_requested=%s "
                         "vision_native_graph_disable_requested=%s\n",
                         __func__, qwen_vl_architecture_name(impl->architecture),
                         impl->backend_name.c_str(),
                         llama_model_n_embd_out(impl->model),
                         llama_model_n_embd_inp(impl->model),
                         impl->deepstack_layer_count, config.expected_image_count,
                         config.image_min_tokens, config.image_max_tokens,
                         llama_n_ctx(impl->context), llama_n_batch(impl->context),
                         llama_n_ubatch(impl->context),
                         config.disable_text_backend_native_graphs ? "true" : "false",
                         config.disable_vision_backend_native_graphs ? "true" : "false");
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
    return std::unique_ptr<Qwen3VLBridge>(new Qwen3VLBridge(std::move(impl)));
}

bool Qwen3VLBridge::extract_token_embeddings(const std::vector<Qwen3VLImageView> & images,
                                              const std::string & instruction, int32_t token_id,
                                              size_t token_count, std::vector<float> & embeddings,
                                              std::string & error) {
    embeddings.clear();
    error.clear();
    if (impl_ == nullptr || impl_->model == nullptr || impl_->context == nullptr ||
        impl_->vision == nullptr || impl_->vocab == nullptr) {
        error = "Qwen3-VL bridge is not initialized";
        return false;
    }
    if (impl_->config.action_token.empty() || impl_->config.action_token_id < 0) {
        error = "Qwen3-VL bridge was configured without an action token";
        return false;
    }
    if (images.size() != static_cast<size_t>(impl_->config.expected_image_count)) {
        error = "Qwen3-VL image count does not match the policy";
        return false;
    }
    if (token_count == 0) {
        error = "Qwen3-VL requested token embedding count must be positive";
        return false;
    }
    if (token_id != impl_->config.action_token_id) {
        error = "Qwen3-VL requested token ID does not match the policy";
        return false;
    }

    try {
        mtmd::input_chunks chunks;
        tokenize_multimodal_prompt(impl_->config, impl_->architecture,
                                   impl_->model, impl_->vision, images,
                                   instruction, chunks);
        PreparedMultimodalBatch prepared =
            prepare_multimodal_batch(impl_->config, impl_->model, impl_->vision, chunks);

        std::vector<size_t> matches;
        for (size_t index = 0; index < prepared.token_ids.size(); ++index) {
            if (prepared.token_ids[index] == token_id) {
                matches.push_back(index);
            }
        }
        if (matches.size() < token_count) {
            throw std::runtime_error("Qwen3-VL prompt contains fewer target tokens than requested");
        }
        matches.erase(matches.begin(), matches.end() - static_cast<std::ptrdiff_t>(token_count));
        if (prepared.token_count > llama_n_batch(impl_->context)) {
            throw std::runtime_error(
                "Qwen3-VL multimodal prompt exceeds n_batch; increase --n-batch for single-batch decode");
        }
        if (prepared.token_count > static_cast<size_t>(llama_n_ctx(impl_->context)) ||
            prepared.position_count > static_cast<llama_pos>(llama_n_ctx(impl_->context))) {
            throw std::runtime_error("Qwen3-VL multimodal prompt exceeds n_ctx");
        }

        const int layer_count = llama_model_n_layer(impl_->model);
        if (layer_count <= 0) {
            throw std::runtime_error("Qwen3-VL model has no decoder layers");
        }
        LayerCapture & capture = impl_->layer_capture;
        capture.enabled = false;
        capture.token_count = prepared.token_count;
        capture.hidden_size = static_cast<size_t>(impl_->config.hidden_size);
        capture.layer_to_slot.assign(static_cast<size_t>(layer_count), -1);
        capture.deepstack_to_slot.assign(static_cast<size_t>(layer_count), -1);
        capture.result_norm_slot = -1;
        if (impl_->architecture == QwenVLArchitecture::qwen2_5_vl) {
            capture.result_norm_slot = 0;
        } else {
            capture.layer_to_slot.back() = 0;
        }
        capture.values.assign(prepared.token_count * capture.hidden_size, 0.0f);
        capture.seen.assign(1, uint8_t{0});
        begin_layer_boundary_tracking(capture, static_cast<size_t>(layer_count));
        capture.error.clear();
        capture.enabled = true;
        std::fill(prepared.outputs.begin(), prepared.outputs.end(), int8_t{1});

        llama_memory_clear(llama_get_memory(impl_->context), true);
        llama_set_embeddings(impl_->context, true);
        llama_batch batch = prepared.view();
        const int32_t decode_result = decode_and_synchronize(impl_->context, batch);
        capture.enabled = false;
        if (decode_result != 0) {
            throw std::runtime_error("failed to evaluate the Qwen3-VL multimodal batch");
        }
        if (!capture.error.empty()) {
            throw std::runtime_error(capture.error);
        }
        std::string boundary_error;
        if (!validate_layer_boundary_tracking(capture, boundary_error)) {
            throw std::runtime_error(boundary_error);
        }
        if (capture.seen.size() != 1 || capture.seen.front() == 0) {
            throw std::runtime_error(
                "Qwen-VL did not expose the final conditioning output");
        }

        embeddings.resize(token_count * capture.hidden_size);
        for (size_t output_index = 0; output_index < matches.size(); ++output_index) {
            const float * hidden =
                capture.values.data() + matches[output_index] * capture.hidden_size;
            std::copy_n(hidden, capture.hidden_size,
                        embeddings.data() + output_index * capture.hidden_size);
        }
        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        impl_->refresh_backend_name();
        return true;
    } catch (const std::exception & exception) {
        llama_synchronize(impl_->context);
        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        llama_memory_clear(llama_get_memory(impl_->context), true);
        embeddings.clear();
        error = exception.what();
        return false;
    }
}

bool Qwen3VLBridge::extract_full_hidden_states(
    const std::vector<Qwen3VLImageView> & images, const std::string & instruction,
    std::vector<float> & hidden_states, std::vector<uint8_t> & attention_mask,
    std::string & error) {
    hidden_states.clear();
    attention_mask.clear();
    error.clear();
    if (impl_ == nullptr || impl_->model == nullptr || impl_->context == nullptr ||
        impl_->vision == nullptr || impl_->vocab == nullptr) {
        error = "Qwen3-VL bridge is not initialized";
        return false;
    }
    if (images.size() != static_cast<size_t>(impl_->config.expected_image_count)) {
        error = "Qwen3-VL image count does not match the policy";
        return false;
    }

    try {
        mtmd::input_chunks chunks;
        tokenize_multimodal_prompt(impl_->config, impl_->architecture,
                                   impl_->model, impl_->vision, images,
                                   instruction, chunks);
        PreparedMultimodalBatch prepared =
            prepare_multimodal_batch(impl_->config, impl_->model, impl_->vision, chunks);
        if (prepared.token_count > llama_n_batch(impl_->context)) {
            throw std::runtime_error(
                "Qwen3-VL multimodal prompt exceeds n_batch; increase --n-batch for single-batch decode");
        }
        if (prepared.token_count > static_cast<size_t>(llama_n_ctx(impl_->context)) ||
            prepared.position_count > static_cast<llama_pos>(llama_n_ctx(impl_->context))) {
            throw std::runtime_error("Qwen3-VL multimodal prompt exceeds n_ctx");
        }
        const size_t hidden_size = static_cast<size_t>(impl_->config.hidden_size);
        if (prepared.token_count > std::numeric_limits<size_t>::max() / hidden_size) {
            throw std::runtime_error("Qwen3-VL hidden-state buffer size overflow");
        }

        const int layer_count = llama_model_n_layer(impl_->model);
        if (layer_count <= 0) {
            throw std::runtime_error("Qwen3-VL model has no decoder layers");
        }
        LayerCapture & capture = impl_->layer_capture;
        capture.enabled = false;
        capture.token_count = prepared.token_count;
        capture.hidden_size = hidden_size;
        capture.layer_to_slot.assign(static_cast<size_t>(layer_count), -1);
        capture.deepstack_to_slot.assign(static_cast<size_t>(layer_count), -1);
        capture.result_norm_slot = -1;
        if (impl_->architecture == QwenVLArchitecture::qwen2_5_vl) {
            capture.result_norm_slot = 0;
        } else {
            capture.layer_to_slot.back() = 0;
        }
        capture.values.assign(prepared.token_count * hidden_size, 0.0f);
        capture.seen.assign(1, uint8_t{0});
        begin_layer_boundary_tracking(capture, static_cast<size_t>(layer_count));
        capture.error.clear();
        capture.enabled = true;
        std::fill(prepared.outputs.begin(), prepared.outputs.end(), int8_t{1});

        llama_memory_clear(llama_get_memory(impl_->context), true);
        llama_set_embeddings(impl_->context, true);
        llama_batch batch = prepared.view();
        const int32_t decode_result = decode_and_synchronize(impl_->context, batch);
        capture.enabled = false;
        if (decode_result != 0) {
            throw std::runtime_error("failed to evaluate the Qwen3-VL multimodal batch");
        }
        if (!capture.error.empty()) {
            throw std::runtime_error(capture.error);
        }
        std::string boundary_error;
        if (!validate_layer_boundary_tracking(capture, boundary_error)) {
            throw std::runtime_error(boundary_error);
        }
        if (capture.seen.size() != 1 || capture.seen.front() == 0) {
            throw std::runtime_error(
                "Qwen-VL did not expose the final conditioning output");
        }

        hidden_states = std::move(capture.values);
        attention_mask.assign(prepared.token_count, uint8_t{1});
        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        impl_->refresh_backend_name();
        return true;
    } catch (const std::exception & exception) {
        llama_synchronize(impl_->context);
        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        llama_memory_clear(llama_get_memory(impl_->context), true);
        hidden_states.clear();
        attention_mask.clear();
        error = exception.what();
        return false;
    }
}

bool Qwen3VLBridge::extract_layer_hidden_states(
    const std::vector<Qwen3VLImageView> & images, const std::string & instruction,
    const std::vector<int32_t> & hidden_tuple_indices, std::vector<float> & hidden_states,
    std::vector<uint8_t> & attention_mask, std::string & error) {
    hidden_states.clear();
    attention_mask.clear();
    error.clear();
    if (impl_ == nullptr || impl_->model == nullptr || impl_->context == nullptr ||
        impl_->vision == nullptr || impl_->vocab == nullptr) {
        error = "Qwen3-VL bridge is not initialized";
        return false;
    }
    if (images.size() != static_cast<size_t>(impl_->config.expected_image_count)) {
        error = "Qwen3-VL image count does not match the policy";
        return false;
    }

    const int model_layer_count = llama_model_n_layer(impl_->model);
    if (hidden_tuple_indices.empty() || model_layer_count <= 0 ||
        hidden_tuple_indices.size() > static_cast<size_t>(model_layer_count)) {
        error = "Qwen3-VL requested hidden-state layer set is incompatible with the model";
        return false;
    }
    std::vector<int> layer_to_slot(static_cast<size_t>(model_layer_count), -1);
    std::vector<int> deepstack_to_slot(static_cast<size_t>(model_layer_count), -1);
    if (impl_->deepstack_layer_count >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "Qwen-VL DeepStack layer count exceeds the supported range";
        return false;
    }
    const int deepstack_layer_count =
        static_cast<int>(impl_->deepstack_layer_count);
    int result_norm_slot = -1;
    for (size_t slot = 0; slot < hidden_tuple_indices.size(); ++slot) {
        const int32_t tuple_index = hidden_tuple_indices[slot];
        QwenVLHiddenStateSource source;
        if (!qwen_vl_hidden_state_source(
                impl_->architecture, model_layer_count, deepstack_layer_count,
                tuple_index, source, error)) {
            return false;
        }
        if (source.kind == QwenVLHiddenStateSourceKind::final_norm) {
            if (result_norm_slot >= 0) {
                error = "Qwen-VL hidden-state tuple indices must be unique";
                return false;
            }
            result_norm_slot = static_cast<int>(slot);
            continue;
        }
        if (source.layer < 0 || source.layer >= model_layer_count) {
            error = "Qwen-VL hidden-state source layer is out of range";
            return false;
        }
        std::vector<int> & target =
            source.kind == QwenVLHiddenStateSourceKind::deepstack_output
                ? deepstack_to_slot
                : layer_to_slot;
        if (target[static_cast<size_t>(source.layer)] >= 0) {
            error = "Qwen-VL hidden-state tuple indices must be unique";
            return false;
        }
        target[static_cast<size_t>(source.layer)] = static_cast<int>(slot);
    }

    try {
        mtmd::input_chunks chunks;
        tokenize_multimodal_prompt(impl_->config, impl_->architecture,
                                   impl_->model, impl_->vision, images,
                                   instruction, chunks);
        PreparedMultimodalBatch prepared =
            prepare_multimodal_batch(impl_->config, impl_->model, impl_->vision, chunks);
        if (prepared.token_count > llama_n_batch(impl_->context)) {
            throw std::runtime_error(
                "Qwen3-VL multimodal prompt exceeds n_batch; increase --n-batch for single-batch decode");
        }
        if (prepared.token_count > static_cast<size_t>(llama_n_ctx(impl_->context)) ||
            prepared.position_count > static_cast<llama_pos>(llama_n_ctx(impl_->context))) {
            throw std::runtime_error("Qwen3-VL multimodal prompt exceeds n_ctx");
        }

        const size_t hidden_size = static_cast<size_t>(impl_->config.hidden_size);
        const size_t requested_layers = hidden_tuple_indices.size();
        if (prepared.token_count > std::numeric_limits<size_t>::max() / hidden_size ||
            prepared.token_count * hidden_size >
                std::numeric_limits<size_t>::max() / requested_layers) {
            throw std::runtime_error("Qwen3-VL layerwise hidden-state buffer size overflow");
        }
        LayerCapture & capture = impl_->layer_capture;
        capture.enabled = false;
        capture.token_count = prepared.token_count;
        capture.hidden_size = hidden_size;
        capture.layer_to_slot = layer_to_slot;
        capture.deepstack_to_slot = deepstack_to_slot;
        capture.result_norm_slot = result_norm_slot;
        capture.values.assign(requested_layers * prepared.token_count * hidden_size, 0.0f);
        capture.seen.assign(requested_layers, uint8_t{0});
        begin_layer_boundary_tracking(capture,
                                      static_cast<size_t>(model_layer_count));
        capture.error.clear();
        capture.enabled = true;
        std::fill(prepared.outputs.begin(), prepared.outputs.end(), int8_t{1});

        llama_memory_clear(llama_get_memory(impl_->context), true);
        llama_set_embeddings(impl_->context, true);
        llama_batch batch = prepared.view();
        const int32_t decode_result = decode_and_synchronize(impl_->context, batch);
        capture.enabled = false;
        if (decode_result != 0) {
            throw std::runtime_error("failed to evaluate the Qwen3-VL multimodal batch");
        }
        if (!capture.error.empty()) {
            throw std::runtime_error(capture.error);
        }
        std::string boundary_error;
        if (!validate_layer_boundary_tracking(capture, boundary_error)) {
            throw std::runtime_error(boundary_error);
        }
        if (std::any_of(capture.seen.begin(), capture.seen.end(),
                        [](uint8_t seen) { return seen == 0; })) {
            throw std::runtime_error(
                "Qwen3-VL did not expose every requested hidden-state output");
        }

        hidden_states = std::move(capture.values);
        attention_mask.assign(prepared.token_count, uint8_t{1});
        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        impl_->refresh_backend_name();
        return true;
    } catch (const std::exception & exception) {
        llama_synchronize(impl_->context);
        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        llama_memory_clear(llama_get_memory(impl_->context), true);
        hidden_states.clear();
        attention_mask.clear();
        error = exception.what();
        return false;
    }
}

bool Qwen3VLBridge::generate_autoregressive(
    const std::vector<Qwen3VLImageView> & images,
    const std::string & instruction,
    const QwenVLGenerationConfig & generation,
    QwenVLGenerationResult & result, std::string & error) {
    result = QwenVLGenerationResult{};
    error.clear();
    if (impl_ == nullptr || impl_->model == nullptr || impl_->context == nullptr ||
        impl_->vision == nullptr || impl_->vocab == nullptr) {
        error = "Qwen-VL bridge is not initialized";
        return false;
    }
    if (images.size() != static_cast<size_t>(impl_->config.expected_image_count)) {
        error = "Qwen-VL image count does not match the policy";
        return false;
    }
    if (generation.max_length == 0 ||
        generation.max_length > static_cast<size_t>(llama_n_ctx(impl_->context)) ||
        generation.max_length > static_cast<size_t>(INT32_MAX) ||
        generation.top_k != 1 || generation.eos_token_ids.empty() ||
        !std::isfinite(generation.repetition_penalty) ||
        generation.repetition_penalty <= 0.0f) {
        error = "Qwen-VL autoregressive generation configuration is incompatible";
        return false;
    }
    std::vector<uint8_t> eos_seen(static_cast<size_t>(impl_->config.vocab_size),
                                  uint8_t{0});
    for (int32_t eos : generation.eos_token_ids) {
        if (eos < 0 || eos >= impl_->config.vocab_size ||
            eos_seen[static_cast<size_t>(eos)] != 0) {
            error = "Qwen-VL generation EOS token set is invalid";
            return false;
        }
        eos_seen[static_cast<size_t>(eos)] = 1;
    }

    try {
        mtmd::input_chunks chunks;
        tokenize_multimodal_prompt(impl_->config, impl_->architecture,
                                   impl_->model, impl_->vision, images,
                                   instruction, chunks);
        PreparedMultimodalBatch prepared =
            prepare_multimodal_batch(impl_->config, impl_->model, impl_->vision,
                                     chunks);
        if (prepared.token_count > llama_n_batch(impl_->context)) {
            throw std::runtime_error(
                "Qwen-VL multimodal prompt exceeds n_batch; increase --n-batch");
        }
        if (prepared.token_count > generation.max_length ||
            prepared.token_count > static_cast<size_t>(llama_n_ctx(impl_->context)) ||
            prepared.position_count >
                static_cast<llama_pos>(llama_n_ctx(impl_->context))) {
            throw std::runtime_error(
                "Qwen-VL multimodal prompt exceeds the FAST max_length/n_ctx contract");
        }

        std::vector<int64_t> input_ids;
        std::vector<uint8_t> attention_mask;
        std::vector<int64_t> image_grid_thw;
        export_prepared_inputs(impl_->config, impl_->vocab, chunks, prepared,
                               input_ids, attention_mask, image_grid_thw);
        if (input_ids.size() != prepared.token_count) {
            throw std::runtime_error(
                "Qwen-VL multimodal prompt token export is inconsistent");
        }
        result.prompt_token_count = prepared.token_count;
        result.full_sequence.reserve(generation.max_length);
        for (int64_t input_id : input_ids) {
            if (input_id < 0 || input_id >= impl_->config.vocab_size) {
                throw std::runtime_error(
                    "Qwen-VL multimodal prompt contains an out-of-vocabulary token");
            }
            result.full_sequence.push_back(static_cast<int32_t>(input_id));
        }
        if (prepared.token_count == generation.max_length) {
            impl_->refresh_backend_name();
            return true;
        }

        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        llama_memory_clear(llama_get_memory(impl_->context), true);
        std::fill(prepared.outputs.begin(), prepared.outputs.end(), int8_t{0});
        prepared.outputs.back() = 1;
        llama_batch prompt_batch = prepared.view();
        if (decode_and_synchronize(impl_->context, prompt_batch) != 0) {
            throw std::runtime_error(
                "failed to evaluate the Qwen-VL autoregressive prompt");
        }

        while (result.full_sequence.size() < generation.max_length) {
            const float * logits = llama_get_logits_ith(impl_->context, -1);
            int32_t next = -1;
            std::string selection_error;
            if (!qwen_vl_select_repetition_penalized_top1(
                    logits, static_cast<size_t>(impl_->config.vocab_size),
                    result.full_sequence, generation.repetition_penalty, next,
                    selection_error)) {
                throw std::runtime_error(selection_error);
            }
            result.full_sequence.push_back(next);
            result.continuation.push_back(next);
            if (eos_seen[static_cast<size_t>(next)] != 0 ||
                result.full_sequence.size() == generation.max_length) {
                break;
            }

            const size_t generation_index = result.continuation.size() - 1U;
            if (generation_index >
                static_cast<size_t>(std::numeric_limits<llama_pos>::max() -
                                    prepared.position_count)) {
                throw std::runtime_error(
                    "Qwen-VL autoregressive M-RoPE position overflow");
            }
            llama_token token = static_cast<llama_token>(next);
            llama_pos position =
                prepared.position_count + static_cast<llama_pos>(generation_index);
            int32_t sequence_count = 1;
            llama_seq_id sequence_value = 0;
            llama_seq_id * sequence = &sequence_value;
            int8_t output = 1;
            llama_batch token_batch{
                1,
                &token,
                nullptr,
                &position,
                &sequence_count,
                &sequence,
                &output,
            };
            if (decode_and_synchronize(impl_->context, token_batch) != 0) {
                throw std::runtime_error(
                    "failed to evaluate an incremental Qwen-VL generation token");
            }
        }

        llama_memory_clear(llama_get_memory(impl_->context), true);
        impl_->refresh_backend_name();
        return true;
    } catch (const std::exception & exception) {
        llama_synchronize(impl_->context);
        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        llama_memory_clear(llama_get_memory(impl_->context), true);
        result = QwenVLGenerationResult{};
        error = exception.what();
        return false;
    }
}

void Qwen3VLBridge::reset() {
    if (impl_ != nullptr && impl_->context != nullptr) {
        llama_synchronize(impl_->context);
        impl_->layer_capture.disable();
        llama_set_embeddings(impl_->context, false);
        llama_memory_clear(llama_get_memory(impl_->context), true);
    }
}

const char * Qwen3VLBridge::backend_name() const {
    if (impl_ == nullptr) {
        return "unknown";
    }
    impl_->refresh_backend_name();
    return impl_->backend_name.c_str();
}

QwenVLArchitecture Qwen3VLBridge::architecture() const {
    return impl_ != nullptr ? impl_->architecture : QwenVLArchitecture::unknown;
}

} // namespace robotcpp::starvla
