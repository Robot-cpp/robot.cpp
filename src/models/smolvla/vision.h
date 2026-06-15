#ifndef SMOLVLA_VISION_H
#define SMOLVLA_VISION_H

#include <cstdint>
#include <string>
#include <vector>

#include "models/ggml_backend.h"

struct gguf_context;
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
typedef struct ggml_gallocr * ggml_gallocr_t;

// SmolVLA Vision context
struct smolvla_vision_ctx {
    // GGUF / GGML runtime
    gguf_context *        ctx_gguf;
    ggml_context *        ctx_data;
    ggml_backend_t        backend_cpu;
    ggml_backend_sched_t  sched;
    std::vector<ggml_backend_t> backends;
    backend_buft_policy   buft_policy;
    std::vector<ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<uint8_t>  buf_compute_meta;
    ggml_cgraph *         graph;
    ggml_tensor *         graph_inp_raw;
    ggml_tensor *         graph_positions;
    ggml_tensor *         graph_vision_output;
    ggml_tensor *         graph_connector_output;
    std::vector<int32_t>  patch_positions;

    // SigLIP config
    int image_size;
    int patch_size;
    int hidden_size;
    int intermediate_size;
    int n_heads;
    int n_layers;
    int num_patches;
    float eps;
    float image_mean[3];
    float image_std[3];
    std::vector<std::string> image_keys;

    // ViT tensors
    ggml_tensor * patch_embd_w;
    ggml_tensor * patch_embd_b;
    ggml_tensor * pos_embd;
    ggml_tensor * post_ln_w;
    ggml_tensor * post_ln_b;

    struct layer_t {
        ggml_tensor * ln1_w; ggml_tensor * ln1_b;
        ggml_tensor * q_w;   ggml_tensor * q_b;
        ggml_tensor * k_w;   ggml_tensor * k_b;
        ggml_tensor * v_w;   ggml_tensor * v_b;
        ggml_tensor * o_w;   ggml_tensor * o_b;
        ggml_tensor * ln2_w; ggml_tensor * ln2_b;
        ggml_tensor * ffn1_w; ggml_tensor * ffn1_b; // ffn_down
        ggml_tensor * ffn2_w; ggml_tensor * ffn2_b; // ffn_up
    };
    std::vector<layer_t> layers;

    // SmolVLA connector (pool + linear) from mm.0.weight in same GGUF
    int pool_num;
    int output_tokens;
    int connector_in_features;
    int connector_out_features;
    int proj_dim;
    ggml_tensor * connector_w;
};

// Load standalone SmolVLA vision model from GGUF
// verbosity: 0=quiet, 1=normal, 2=debug
smolvla_vision_ctx * smolvla_vision_load(const char * mmproj_path, int verbosity = 1);

// Free vision context
void smolvla_vision_free(smolvla_vision_ctx * ctx);

// Encode image from file path
// Returns embedding vector of size [num_output_tokens * proj_dim]
// For SmolVLA: 64 * 960 = 61440 floats
std::vector<float> smolvla_vision_encode_file(
    smolvla_vision_ctx * ctx,
    const char * image_path,
    int n_threads = 4
);

// Encode image from in-memory encoded bytes (JPEG/PNG)
std::vector<float> smolvla_vision_encode_bytes(
    smolvla_vision_ctx * ctx,
    const unsigned char * image_bytes,
    int image_len,
    int n_threads = 4
);

// Encode image from raw RGB data (HWC format, uint8)
std::vector<float> smolvla_vision_encode_raw(
    smolvla_vision_ctx * ctx,
    const uint8_t * data,
    int width,
    int height,
    int channels,
    int stride_bytes,
    int n_threads = 4
);

// Get embedding dimensions
int smolvla_vision_embd_size(const smolvla_vision_ctx * ctx);  // Total embedding size
int smolvla_vision_n_tokens(const smolvla_vision_ctx * ctx);   // Number of output tokens

#endif // SMOLVLA_VISION_H
