#pragma once

#include "models/model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp::starvla {

enum class StarVLAVariant {
    qwen3_oft,
    qwen3_groot,
    qwen3_pi_v3,
    qwen25_oft,
    qwen25_groot,
    qwen25_pi,
    qwen25_fast,
};

const char * starvla_variant_name(StarVLAVariant variant) noexcept;
const char * starvla_variant_framework(StarVLAVariant variant) noexcept;
bool starvla_variant_from_metadata(const std::string & framework,
                                   const std::string & backbone,
                                   StarVLAVariant & variant) noexcept;

struct StarVLAEngineConfig {
    std::string policy_path;
    std::string text_path_override;
    std::string mmproj_path_override;
    std::string unnorm_key;
    int n_threads = 0;
    int n_ctx = 2048;
    int n_batch = 512;
    int64_t noise_seed = -1;
    int verbosity = 0;
};

struct StarVLAStageTimings {
    double image_preprocess_ms = 0.0;
    double prompt_ms = 0.0;
    double qwen3vl_ms = 0.0;
    double policy_ms = 0.0;
    double unnormalize_ms = 0.0;
    double total_ms = 0.0;
};

struct StarVLAEngineResult {
    std::vector<float> actions;
    std::vector<float> normalized_actions;
    std::vector<float> action_queries;
    std::vector<int32_t> generated_token_ids;
    std::vector<int32_t> action_token_ids;
    std::vector<int32_t> fast_token_ids;
    std::string instruction;
    std::string unnorm_key_used;
    int chunk_size = 0;
    int action_dim = 0;
    StarVLAStageTimings timings;
};

class StarVLAEngine {
  public:
    ~StarVLAEngine();

    StarVLAEngine(const StarVLAEngine &) = delete;
    StarVLAEngine & operator=(const StarVLAEngine &) = delete;

    static std::unique_ptr<StarVLAEngine> load(const StarVLAEngineConfig & config,
                                               std::string & error);

    bool predict(const observation & obs, StarVLAEngineResult & result, std::string & error);
    bool predict_with_noise(const observation & obs, const std::vector<float> & initial_noise,
                            StarVLAEngineResult & result, std::string & error);
    void reset();

  private:
    struct Impl;

    explicit StarVLAEngine(std::unique_ptr<Impl> impl);
    bool predict_impl(const observation & obs, const std::vector<float> * initial_noise,
                      StarVLAEngineResult & result, std::string & error);

    std::unique_ptr<Impl> impl_;
};

} // namespace robotcpp::starvla
