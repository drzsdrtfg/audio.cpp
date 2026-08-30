#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_frontend.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/smart_turn/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::smart_turn {

struct SmartTurnInferenceResult {
    float probability = 0.0F;
};

struct SmartTurnHeadWeights {
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

    // Runs the encoder + attention-pool + classifier on log-mel features.
    SmartTurnInferenceResult infer_features(const std::vector<float> & mel_features);

    SmartTurnInferenceResult infer_audio(const runtime::AudioBuffer & audio);

    const SmartTurnConfig & config() const noexcept { return weights_->config; }

private:
    struct HeadGraph;
    HeadGraph & ensure_head_graph();

    std::shared_ptr<const SmartTurnWeights> weights_;
    std::shared_ptr<const modules::WhisperFrontendComponentWeights> frontend_weights_;
    modules::WhisperFrontendComponent frontend_;
    SmartTurnHeadWeights head_weights_;
    engine::audio::WhisperLogMelExtractor extractor_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::unique_ptr<HeadGraph> head_graph_;
};

}  // namespace engine::models::smart_turn
