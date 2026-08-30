#include "engine/models/smart_turn/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/weight_metadata.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::smart_turn {
namespace io = engine::io;

SmartTurnAssetPaths resolve_smart_turn_assets(const std::filesystem::path & model_path) {
    std::filesystem::path checkpoint_path = model_path;
    if (!engine::io::is_existing_file(checkpoint_path)) {
        if (!engine::io::is_existing_directory(model_path)) {
            throw std::runtime_error("Smart Turn model path does not exist: " + model_path.string());
        }
        checkpoint_path = model_path / "smart_turn.safetensors";
        if (!engine::io::is_existing_file(checkpoint_path)) {
            throw std::runtime_error("Smart Turn weights not found: " + checkpoint_path.string());
        }
    }

    assets::ResourceBundle resources(checkpoint_path.parent_path());
    resources.add_file("weights", checkpoint_path);
    resources.add_file("config", assets::checkpoint_sidecar_config_path(checkpoint_path));
    (void) resources.parse_json("config");

    SmartTurnAssetPaths paths;
    paths.model_root = resources.model_root();
    paths.checkpoint_path = resources.require_file("weights");
    paths.config_path = resources.require_file("config");
    return paths;
}

namespace {

SmartTurnConfig load_smart_turn_config(const io::json::Value & root) {
    SmartTurnConfig cfg;
    cfg.sample_rate = root.require("sample_rate").as_i64();
    cfg.chunk_samples = root.require("chunk_samples").as_i64();
    cfg.n_mels = root.require("n_mels").as_i64();
    cfg.n_fft = root.require("n_fft").as_i64();
    cfg.hop_length = root.require("hop_length").as_i64();
    cfg.mel_frames = root.require("mel_frames").as_i64();
    cfg.n_audio_ctx = root.require("n_audio_ctx").as_i64();
    cfg.d_model = root.require("d_model").as_i64();
    cfg.n_audio_head = root.require("n_audio_head").as_i64();
    cfg.n_audio_layer = root.require("n_audio_layer").as_i64();
    cfg.ffn_dim = root.require("ffn_dim").as_i64();
    if (const auto it = root.find("pool_dim"); it != nullptr && it->is_number()) {
        cfg.pool_dim = it->as_i64();
    }
    if (const auto it = root.find("classifier_hidden"); it != nullptr && it->is_number()) {
        cfg.classifier_hidden = it->as_i64();
    }
    if (const auto it = root.find("classifier_mid"); it != nullptr && it->is_number()) {
        cfg.classifier_mid = it->as_i64();
    }
    if (const auto it = root.find("layer_norm_eps"); it != nullptr && it->is_number()) {
        cfg.layer_norm_eps = static_cast<float>(it->as_number());
    }
    if (const auto it = root.find("threshold"); it != nullptr && it->is_number()) {
        cfg.threshold = static_cast<float>(it->as_number());
    }

    if (cfg.sample_rate <= 0 || cfg.chunk_samples <= 0 || cfg.n_mels <= 0 || cfg.n_fft <= 0 ||
        cfg.hop_length <= 0 || cfg.mel_frames <= 0 || cfg.n_audio_ctx <= 0 || cfg.d_model <= 0 ||
        cfg.n_audio_head <= 0 || cfg.n_audio_layer <= 0 || cfg.ffn_dim <= 0) {
        throw std::runtime_error("Smart Turn config contains invalid dimensions");
    }
    if (cfg.d_model % cfg.n_audio_head != 0) {
        throw std::runtime_error("Smart Turn d_model must be divisible by n_audio_head");
    }
    if (cfg.mel_frames != cfg.n_audio_ctx * 2) {
        throw std::runtime_error("Smart Turn mel_frames must equal n_audio_ctx * 2");
    }
    if (cfg.chunk_samples != cfg.mel_frames * cfg.hop_length) {
        throw std::runtime_error("Smart Turn chunk_samples must equal mel_frames * hop_length");
    }
    if (cfg.threshold < 0.0F || cfg.threshold > 1.0F) {
        throw std::runtime_error("Smart Turn threshold must be within [0, 1]");
    }
    return cfg;
}

SmartTurnWeights load_smart_turn_weights(const std::filesystem::path & checkpoint_path) {
    const auto paths = resolve_smart_turn_assets(checkpoint_path);
    assets::ResourceBundle resources(paths.model_root);
    resources.add_file("weights", paths.checkpoint_path);
    resources.add_file("config", paths.config_path);

    SmartTurnWeights weights;
    weights.config = load_smart_turn_config(resources.parse_json("config"));
    weights.source = resources.open_tensor_source("weights");

    // Basic structural validation against the tensor manifest.
    const auto & source = *weights.source;
    if (!source.has_tensor("model.encoder.conv1.weight") ||
        !source.has_tensor("model.encoder.embed_positions.weight") ||
        !source.has_tensor("pool_attention.0.weight") ||
        !source.has_tensor("classifier.6.weight")) {
        throw std::runtime_error("Smart Turn weights are missing required tensors");
    }
    const auto & positions = source.require_metadata("model.encoder.embed_positions.weight");
    if (static_cast<int64_t>(positions.shape.size()) != 2 ||
        positions.shape.at(0) != weights.config.n_audio_ctx ||
        positions.shape.at(1) != weights.config.d_model) {
        throw std::runtime_error("Smart Turn positional embedding shape mismatch");
    }
    return weights;
}

std::string checkpoint_cache_key(const std::filesystem::path & checkpoint_path) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(checkpoint_path, ec);
    return ec ? checkpoint_path.lexically_normal().string() : canonical.string();
}

}  // namespace

std::shared_ptr<const SmartTurnWeights> load_smart_turn_weights_cached(
    const std::filesystem::path & checkpoint_path) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<const SmartTurnWeights>> cache;
    const auto key = checkpoint_cache_key(checkpoint_path);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(key); it != cache.end()) {
            if (auto existing = it->second.lock()) {
                return existing;
            }
        }
    }
    auto loaded = std::make_shared<const SmartTurnWeights>(load_smart_turn_weights(checkpoint_path));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[key] = loaded;
    }
    return loaded;
}

}  // namespace engine::models::smart_turn
