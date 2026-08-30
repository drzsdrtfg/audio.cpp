#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::smart_turn {

// Smart Turn v3 (pipecat-ai): Whisper-Tiny encoder + attention-pool + classifier.
// Predicts whether an 8-second utterance ends a complete turn.
struct SmartTurnConfig {
    int64_t sample_rate = 16000;
    int64_t chunk_samples = 128000;
    int64_t n_mels = 80;
    int64_t n_fft = 400;
    int64_t hop_length = 160;
    int64_t mel_frames = 800;
    int64_t n_audio_ctx = 400;
    int64_t d_model = 384;
    int64_t n_audio_head = 6;
    int64_t n_audio_layer = 4;
    int64_t ffn_dim = 1536;
    int64_t pool_dim = 256;
    int64_t classifier_hidden = 256;
    int64_t classifier_mid = 64;
    float layer_norm_eps = 1.0e-5F;
    float threshold = 0.5F;
};

struct SmartTurnWeights {
    SmartTurnConfig config;
    std::shared_ptr<const assets::TensorSource> source;
};

struct SmartTurnAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path checkpoint_path;
    std::filesystem::path config_path;
};

SmartTurnAssetPaths resolve_smart_turn_assets(const std::filesystem::path & model_path);
std::shared_ptr<const SmartTurnWeights> load_smart_turn_weights_cached(
    const std::filesystem::path & checkpoint_path);

}  // namespace engine::models::smart_turn
