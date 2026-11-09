#include "bitvla_model.h"
#include "component.h"
#include "ops.h"
#include "models/image_preprocess.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

namespace robotcpp {

namespace bitvla {
std::unique_ptr<float[]> prompt_input_without_images(const std::vector<float> & prompt,
                                                     const std::vector<size_t> & image_positions) {
    constexpr size_t width = 2560;
    if (prompt.size() % width)
        throw std::runtime_error("Invalid BitVLA prompt row length");
    const size_t rows = prompt.size() / width;
    auto input        = std::unique_ptr<float[]>(new float[prompt.size()]);
    size_t cursor     = 0;
    for (const size_t row : image_positions) {
        if (row < cursor || row >= rows)
            throw std::runtime_error("Invalid BitVLA image positions");
        if (row > cursor)
            std::copy_n(prompt.data() + cursor * width, (row - cursor) * width, input.get() + cursor * width);
        cursor = row + 1;
    }
    if (cursor < rows)
        std::copy_n(prompt.data() + cursor * width, (rows - cursor) * width, input.get() + cursor * width);
    // Every excluded row is replaced by this predict's image features before any read.
    // State/action/text rows retain the template's exact original bytes. No shared mutable buffer.
    return input;
}
} // namespace bitvla

namespace {
using clock = std::chrono::steady_clock;
double milliseconds(clock::time_point start, clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class BitVLAModel final : public Model {
  public:
    explicit BitVLAModel(const model_args & args) {
        llama_backend_init();
        const int threads = args.threads > 0 ? args.threads : std::max(1u, std::thread::hardware_concurrency());
        language.load(args.llm_path, threads, args.verbosity);
        vision.load(args.mmproj_path, threads, args.verbosity);
        proprio.load(args.state_proj_path, threads, args.verbosity);
        action.load(args.action_expert_path, threads, args.verbosity);
        llama_model_params params = llama_model_default_params();
        params.vocab_only         = true;
        tokenizer.reset(llama_model_load_from_file(args.llm_path.c_str(), params));
        if (!tokenizer)
            throw std::runtime_error("BitVLA tokenizer load failed");
        vocab         = llama_model_get_vocab(tokenizer.get());
        image_token   = tokenize("<|image_pad|>", false).at(0);
        proprio_token = tokenize("<proprio_pad>", false).at(0);
        stop_token    = tokenize("<|end_of_text|>", false).at(0);
        build_vision(vision);
        build_proprio(proprio);
        build_action(action);
    }

    const char * type() const override { return "bitvla"; }

    bool predict(const observation & obs, model_result & out, std::string & error) override {
        out = {};
        if (obs.images.empty() || obs.images.size() > 2 || obs.state.size() != 8 || obs.task.empty()) {
            error = "BitVLA requires one or two images, an 8D state, and a task";
            return false;
        }
        for (const auto & image : obs.images) {
            if (!image.data || image.width <= 0 || image.height <= 0 ||
                image.width > std::numeric_limits<int>::max() / 3 || image.channels != 3 ||
                image.stride_bytes < image.width * 3) {
                error = "BitVLA expects RGB images with positive dimensions and a valid row stride";
                return false;
            }
        }
        try {
            const auto start = clock::now();
            if (obs.task != task || obs.images.size() != prompt_image_count)
                prepare_prompt(obs.task, obs.images.size());
            const bool split_host_state_input = bitvla::uses_split_host_state_input(vision, language, action);
            const auto embeddings = split_host_state_input
                                        ? std::unique_ptr<float[]>()
                                        : bitvla::prompt_input_without_images(prompt_embeddings, image_positions);
            size_t image_index    = 0;
            const int image_size  = vision.integer("clip.vision.image_size");
            std::vector<uint8_t> resized;
            for (const auto & image : obs.images) {
                const uint8_t * rgb = image.data;
                int stride          = image.stride_bytes;
                if (image.width != image_size || image.height != image_size) {
                    if (!resize_pillow_bicubic_rgb(rgb, image.width, image.height, stride, image_size, image_size,
                                                   resized, error)) {
                        throw std::runtime_error(error);
                    }
                    rgb    = resized.data();
                    stride = image_size * 3;
                }
                bitvla::encode_vision_for_language(vision, language, rgb, stride, embeddings.get(), image_positions,
                                                   image_index);
            }
            const auto after_vision    = clock::now();
            bool reusable_state_output = split_host_state_input;
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
            // CUDA215 adaptation: retain CPU host embeddings and reuse only state output.
            reusable_state_output =
                reusable_state_output || (vision.is_cpu() && language.is_cpu() && proprio.is_cpu() && action.is_cpu() &&
                                          proprio.output && proprio.output->type == GGML_TYPE_F32 &&
                                          ggml_is_contiguous(proprio.output) && ggml_nelements(proprio.output) == 2560);
#endif
            const auto state_features =
                reusable_state_output ? std::vector<float>() : bitvla::encode_proprio(proprio, obs.state);
            if (reusable_state_output)
                bitvla::encode_proprio_to_host(proprio, obs.state, state_features_host);
            if (!split_host_state_input) {
                const auto & current_state = reusable_state_output ? state_features_host : state_features;
                std::copy(current_state.begin(), current_state.end(), embeddings.get() + proprio_position * 2560);
            }
            const auto after_state = clock::now();
            const auto hidden =
                language.compute_for_action(vision, action, split_host_state_input, prompt_embeddings, embeddings.get(),
                                            image_positions, proprio_position, state_features_host);
            const auto after_language = clock::now();
            out.actions               = bitvla::decode_language_output(action, hidden);
            const auto end            = clock::now();
            out.chunk_size            = 8;
            out.action_dim            = 7;
            out.metrics               = {{"vision_ms", milliseconds(start, after_vision)},
                                         {"state_proj_ms", milliseconds(after_vision, after_state)},
                                         {"llm_ms", milliseconds(after_state, after_language)},
                                         {"action_head_ms", milliseconds(after_language, end)},
                                         {"model_total_ms", milliseconds(start, end)}};
            return true;
        } catch (const std::exception & exception) {
            language.invalidate_static_input();
            error = exception.what();
            return false;
        }
    }

  private:
    std::vector<llama_token> tokenize(const std::string & text, bool special) const {
        const int count = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, special, true);
        std::vector<llama_token> result(count);
        const int actual = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), special, true);
        if (actual < 0)
            throw std::runtime_error("BitVLA tokenization failed");
        result.resize(actual);
        return result;
    }

    void embedding(llama_token token, float * output) {
        auto * weight       = language.tensor("token_embd.weight");
        const size_t offset = token * weight->nb[1];
        if (weight->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(weight, output, offset, 2560 * sizeof(float));
        } else if (weight->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> row(2560);
            ggml_backend_tensor_get(weight, row.data(), offset, row.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < row.size(); ++i)
                output[i] = ggml_fp16_to_fp32(row[i]);
        } else if (weight->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> row(2560);
            ggml_backend_tensor_get(weight, row.data(), offset, row.size() * sizeof(ggml_bf16_t));
            for (size_t i = 0; i < row.size(); ++i)
                output[i] = ggml_bf16_to_fp32(row[i]);
        } else {
            throw std::runtime_error("BitVLA token embeddings require F32, F16, or BF16 storage");
        }
    }

    void prepare_prompt(const std::string & instruction, size_t image_count) {
        task.clear();
        language.invalidate_static_input();
        std::string lowered = instruction;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        std::string text = "System: A chat between a curious human and an artificial intelligence assistant. "
                           "The assistant gives helpful, detailed, and polite answers to the human's questions."
                           "<|eot_id|>User: ";
        for (size_t image = 0; image < image_count; ++image) {
            for (int patch = 0; patch < 256; ++patch)
                text += "<|image_pad|>";
        }
        text += "<proprio_pad>What action should the robot take to " + lowered + "?<|eot_id|>Assistant: ";
        auto tokens            = tokenize(text, true);
        const int action_start = tokens.size();
        tokens.resize(tokens.size() + 56, 1);
        tokens.push_back(stop_token);
        prompt_embeddings.assign(tokens.size() * 2560, 0.0f);
        image_positions.clear();
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] == image_token)
                image_positions.push_back(i);
            else if (tokens[i] == proprio_token)
                proprio_position = i;
            else if (i < static_cast<size_t>(action_start) || i == tokens.size() - 1)
                embedding(tokens[i], prompt_embeddings.data() + i * 2560);
        }
        if (image_positions.size() != image_count * 256)
            throw std::runtime_error("BitVLA tokenizer must expand each image to 256 tokens");
        language.prepare_language_graph(tokens.size(), action_start);
        task               = instruction;
        prompt_image_count = image_count;
    }

    bitvla::component language, vision, proprio, action;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> tokenizer{nullptr, llama_model_free};
    const llama_vocab * vocab = nullptr;
    llama_token image_token = 0, proprio_token = 0, stop_token = 0;
    std::string task;
    size_t prompt_image_count = 0;
    std::vector<float> prompt_embeddings;
    std::vector<size_t> image_positions;
    size_t proprio_position                = 0;
    std::vector<float> state_features_host = std::vector<float>(2560);
};
} // namespace

bool make_bitvla_model(const model_args & args, std::unique_ptr<Model> & out, std::string & error) {
    try {
        out = std::make_unique<BitVLAModel>(args);
        return true;
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
}
} // namespace robotcpp
