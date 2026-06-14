#pragma once

#include "models/pi0/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;

namespace robotcpp::pi0 {

struct Pi0Context;

class Pi0Tokenizer {
public:
    explicit Pi0Tokenizer(const std::string & tokenizer_path);
    ~Pi0Tokenizer();

    Pi0Tokenizer(const Pi0Tokenizer &) = delete;
    Pi0Tokenizer & operator=(const Pi0Tokenizer &) = delete;

    bool available() const;
    bool tokenize_prompt(const std::string & prompt, int max_tokens, std::vector<int32_t> & out) const;

private:
    struct LlamaModelDeleter {
        void operator()(llama_model * model) const;
    };

    std::unique_ptr<llama_model, LlamaModelDeleter> model_;
    const llama_vocab * vocab_ = nullptr;
};

bool pi0_has_vision_encoder(const Pi0Context & ctx);
void pi0_encode_vision(
    const Pi0Context & ctx,
    const std::vector<Pi0ImageTensor> & images,
    std::vector<float> & out,
    int & token_count);
bool pi0_has_language_layer(const Pi0Context & ctx, int layer);
void pi0_prefill_language_prefix_batch(
    const Pi0Context & ctx,
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out,
    uint64_t generation,
    bool need_output = true);
bool pi0_has_merger(const Pi0Context & ctx);
bool pi0_has_vision_prefix(const Pi0Context & ctx);
bool pi0_has_language_prefix(const Pi0Context & ctx);
void pi0_embed_prompt(
    const Pi0Context & ctx,
    const std::string & prompt,
    std::vector<float> & out,
    int & token_count);
void pi0_embed_prompt_tokens(
    const Pi0Context & ctx,
    const std::vector<int32_t> & tokens,
    std::vector<float> & out,
    int & token_count);
void pi0_prefill_prefix(const Pi0Context & ctx, Pi0KvCache & cache, const Pi0Observation & observation);

} // namespace robotcpp::pi0
