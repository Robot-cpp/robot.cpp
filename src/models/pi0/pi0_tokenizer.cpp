#include "models/pi0/pi0_tokenizer.h"

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <fstream>

namespace vlacpp {
namespace {

std::string clean_prompt(const std::string & prompt) {
    size_t begin = 0;
    size_t end = prompt.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(prompt[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(prompt[end - 1]))) {
        --end;
    }

    std::string cleaned;
    cleaned.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        char ch = prompt[i];
        cleaned.push_back(ch == '_' || ch == '\n' ? ' ' : ch);
    }
    return cleaned;
}

bool tokenize_with_vocab(
    const llama_vocab * vocab,
    const std::string & text,
    bool add_special,
    std::vector<int32_t> & out) {
    int32_t count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        add_special,
        false);
    if (count < 0) {
        count = -count;
    }
    if (count <= 0) {
        return true;
    }
    const size_t offset = out.size();
    out.resize(offset + static_cast<size_t>(count));
    const int32_t written = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        reinterpret_cast<llama_token *>(out.data() + offset),
        count,
        add_special,
        false);
    if (written < 0) {
        out.resize(offset);
        return false;
    }
    out.resize(offset + static_cast<size_t>(written));
    return true;
}

std::vector<std::string> tokenizer_candidates(const std::string & model_path) {
    std::vector<std::string> candidates;
    const size_t dot = model_path.find_last_of('.');
    if (dot != std::string::npos) {
        candidates.push_back(model_path.substr(0, dot) + ".tokenizer.gguf");
    }
    candidates.push_back(model_path + ".tokenizer.gguf");
    candidates.push_back(model_path);
    return candidates;
}

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return static_cast<bool>(file);
}

} // namespace

Pi0Tokenizer::Pi0Tokenizer(const std::string & model_path) {
    if (model_path.empty()) {
        return;
    }
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.use_mmap = true;
    for (const std::string & candidate : tokenizer_candidates(model_path)) {
        if (!file_exists(candidate)) {
            continue;
        }
        model_.reset(llama_model_load_from_file(candidate.c_str(), params));
        if (model_ != nullptr) {
            vocab_ = llama_model_get_vocab(model_.get());
            return;
        }
    }
}

Pi0Tokenizer::~Pi0Tokenizer() = default;

void Pi0Tokenizer::LlamaModelDeleter::operator()(llama_model * model) const {
    if (model != nullptr) {
        llama_model_free(model);
    }
}

bool Pi0Tokenizer::available() const {
    return vocab_ != nullptr;
}

bool Pi0Tokenizer::tokenize_prompt(const std::string & prompt, int max_tokens, std::vector<int32_t> & out) const {
    out.clear();
    if (!available() || max_tokens <= 0) {
        return false;
    }
    const std::string cleaned = clean_prompt(prompt) + "\n";
    if (!tokenize_with_vocab(vocab_, cleaned, true, out)) {
        out.clear();
        return false;
    }
    if (static_cast<int>(out.size()) > max_tokens) {
        out.resize(static_cast<size_t>(max_tokens));
    }
    return true;
}

} // namespace vlacpp
