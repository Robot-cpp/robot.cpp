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
    std::string text_path;
    std::string mmproj_path;
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
    void reset();

  private:
    struct Impl;

    explicit StarVLAEngine(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace robotcpp::starvla
