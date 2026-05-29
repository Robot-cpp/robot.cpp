#pragma once

#include "llama.h"
#include "llama-model.h"

#define private public
#include "llama-kv-cache.h"
#undef private

static inline ggml_tensor * llama_get_model_tensor(llama_model * model, const char * name) {
    return const_cast<ggml_tensor *>(model ? model->get_tensor(name) : nullptr);
}

static inline bool llama_kv_cache_is_v_trans(llama_context * ctx) {
    llama_memory_t mem = ctx ? llama_get_memory(ctx) : nullptr;
    llama_kv_cache * kv = static_cast<llama_kv_cache *>(mem);
    return kv ? kv->v_trans : false;
}

static inline ggml_tensor * llama_kv_cache_k_tensor(llama_context * ctx, int32_t il) {
    llama_memory_t mem = ctx ? llama_get_memory(ctx) : nullptr;
    llama_kv_cache * kv = static_cast<llama_kv_cache *>(mem);
    return kv && il >= 0 && il < (int32_t) kv->layers.size() ? kv->layers[il].k : nullptr;
}

static inline ggml_tensor * llama_kv_cache_v_tensor(llama_context * ctx, int32_t il) {
    llama_memory_t mem = ctx ? llama_get_memory(ctx) : nullptr;
    llama_kv_cache * kv = static_cast<llama_kv_cache *>(mem);
    return kv && il >= 0 && il < (int32_t) kv->layers.size() ? kv->layers[il].v : nullptr;
}

static inline void llama_kv_cache_clear(llama_context * ctx) {
    if (ctx) {
        llama_memory_clear(llama_get_memory(ctx), true);
    }
}
