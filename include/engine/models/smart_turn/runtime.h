#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/smart_turn/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::smart_turn {

struct SmartTurnInferenceResult {
    float probability = 0.0F;
};

// Encoder + head weights that live in one backend weight store.
struct SmartTurnBackendWeights {
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    modules::WhisperEmbeddingWeights encoder;
    modules::LinearWeights pool_attention_0;   // d_model -> pool_dim
    modules::LinearWeights pool_attention_2;   // pool_dim -> 1
    modules::LinearWeights classifier_0;       // d_model -> classifier_hidden
    modules::NormWeights classifier_1;         // LayerNorm(classifier_hidden)
    modules::LinearWeights classifier_4;       // classifier_hidden -> classifier_mid
    modules::LinearWeights classifier_6;       // classifier_mid -> 1
};

class SmartTurnRuntime {
public:
    SmartTurnRuntime(
        std::shared_ptr<const SmartTurnWeights> weights,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_storage_type);
    ~SmartTurnRuntime();

    SmartTurnRuntime(const SmartTurnRuntime &) = delete;
    SmartTurnRuntime & operator=(const SmartTurnRuntime &) = delete;

    // Log-mel features (n_mels * mel_frames) for the analyzed 8-second window.
    std::vector<float> extract_audio_features(const runtime::AudioBuffer & audio) const;

    // Log-mel over an already normalized 8-second window; splits the frame range
    // across worker threads (the vendored FFT is single-threaded per call).
    std::vector<float> compute_log_mel(const std::vector<float> & window) const;

    // Runs the fused encoder + attention-pool + classifier graph on log-mel features.
    SmartTurnInferenceResult infer_features(const std::vector<float> & mel_features);

    SmartTurnInferenceResult infer_audio(const runtime::AudioBuffer & audio);

    const SmartTurnConfig & config() const noexcept { return weights_->config; }

private:
    struct InferenceGraph;
    InferenceGraph & ensure_inference_graph();

    std::shared_ptr<const SmartTurnWeights> weights_;
    std::shared_ptr<const SmartTurnBackendWeights> backend_weights_;
    modules::WhisperEmbeddingConfig encoder_config_;
    engine::audio::WhisperLogMelExtractor extractor_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::unique_ptr<InferenceGraph> inference_graph_;
};

}  // namespace engine::models::smart_turn
