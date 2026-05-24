#pragma once

#include "models/model.h"

namespace vlacpp {

class Pi0LanguagePrefix {
public:
    Pi0LanguagePrefix(const ModelConfig & config, const BackendConfig & backend, const TensorMap & tensors);

    bool has_layer(int layer) const;
    void prefill_batch(
        const std::vector<float> & tokens,
        const std::vector<int> & positions,
        int batch,
        int heads,
        int kv_heads,
        int head_dim,
        std::vector<PrefixLayerKv> & cache,
        std::vector<float> & out,
        uint64_t generation = 0,
        bool need_output = true) const;

private:
    bool prefill_layers_batch(
        const std::vector<float> & tokens,
        const std::vector<int> & positions,
        int batch,
        int heads,
        int kv_heads,
        int head_dim,
        std::vector<PrefixLayerKv> & cache,
        std::vector<float> & out,
        uint64_t generation,
        bool need_output) const;
    bool prefill_layer_batch(
        int layer,
        const std::vector<float> & tokens,
        const std::vector<int> & positions,
        int batch,
        int heads,
        int kv_heads,
        int head_dim,
        PrefixLayerKv & layer_cache,
        std::vector<float> & out) const;
    void norm_batch(
        int layer,
        const char * weight_name,
        const std::vector<float> & tokens,
        int batch,
        std::vector<float> & out) const;
    void final_norm_batch(const std::vector<float> & tokens, int batch, std::vector<float> & out) const;
    void qkv_batch(
        int layer,
        const std::vector<float> & tokens,
        int batch,
        std::vector<float> & q,
        std::vector<float> & k,
        std::vector<float> & v) const;
    void rope_batch(
        const std::vector<float> & values,
        const std::vector<int> & positions,
        int tokens,
        int heads,
        int head_dim,
        std::vector<float> & out) const;
    void self_attention_batch(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        int tokens,
        int heads,
        int kv_heads,
        int head_dim,
        std::vector<float> & out) const;
    void attention_out_batch(int layer, const std::vector<float> & values, int batch, std::vector<float> & out) const;
    void mlp_batch(int layer, const std::vector<float> & tokens, int batch, std::vector<float> & out) const;
    const Tensor * find_tensor(const std::string & name) const;
    std::string layer_prefix(int layer) const;

    const ModelConfig & config_;
    const BackendConfig & backend_;
    const TensorMap & tensors_;
};

} // namespace vlacpp
