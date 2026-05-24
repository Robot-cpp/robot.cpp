#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;

namespace vlacpp {

class Pi0Tokenizer {
public:
    explicit Pi0Tokenizer(const std::string & model_path);
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

} // namespace vlacpp
