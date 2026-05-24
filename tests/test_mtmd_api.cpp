#include "clip.h"
#include "mtmd.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main() {
    const char * marker = mtmd_default_marker();
    if (marker == nullptr || std::string(marker) != "<__media__>") {
        std::cerr << "unexpected mtmd media marker\n";
        return 1;
    }

    const mtmd_context_params params = mtmd_context_params_default();
    if (params.media_marker == nullptr || std::string(params.media_marker) != "<__media__>") {
        std::cerr << "unexpected mtmd context media marker\n";
        return 1;
    }
    if (params.n_threads <= 0) {
        std::cerr << "mtmd default n_threads should be positive\n";
        return 1;
    }
    if (params.image_min_tokens != -1 || params.image_max_tokens != -1) {
        std::cerr << "mtmd dynamic image token defaults changed\n";
        return 1;
    }
    const char * model_path = std::getenv("VLACPP_MTMD_MODEL");
    if (model_path != nullptr && std::string(model_path).size() > 0) {
        clip_context_params load_params{};
        load_params.use_gpu = false;
        load_params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        load_params.image_min_tokens = -1;
        load_params.image_max_tokens = -1;
        load_params.warmup = false;
        clip_init_result result = clip_init(model_path, load_params);
        if (result.ctx_v == nullptr) {
            std::cerr << "failed to load mtmd vision model\n";
            return 1;
        }
        if (!clip_has_vision_encoder(result.ctx_v) || clip_n_mmproj_embd(result.ctx_v) <= 0) {
            std::cerr << "expected mtmd vision support\n";
            clip_free(result.ctx_v);
            if (result.ctx_a != nullptr) {
                clip_free(result.ctx_a);
            }
            return 1;
        }
        clip_free(result.ctx_v);
        if (result.ctx_a != nullptr) {
            clip_free(result.ctx_a);
        }
    }
    return 0;
}
