#include "engine/models/smart_turn/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::smart_turn {
namespace {

std::filesystem::path resolve_weight_path(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("Smart Turn model path does not exist: " + model_path.string());
    }
    const auto candidate = model_path / "smart_turn.safetensors";
    if (!engine::io::is_existing_file(candidate)) {
        throw std::runtime_error("Smart Turn weights not found: " + candidate.string());
    }
    return std::filesystem::weakly_canonical(candidate);
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    if (engine::io::is_existing_file(request.model_path)) {
        return {{"default", std::filesystem::weakly_canonical(request.model_path)}};
    }
    return runtime::discover_named_assets(
        std::filesystem::weakly_canonical(request.model_path),
        {"smart_turn.safetensors"});
}

assets::TensorStorageType smart_turn_weight_type_from_options(const runtime::SessionOptions & options) {
    const auto it = options.options.find("smart_turn.weight_type");
    if (it == options.options.end()) {
        return assets::TensorStorageType::Native;
    }
    const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error("smart_turn.weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

class SmartTurnLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "smart_turn";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline}},
        };
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) resolve_smart_turn_assets(resolve_weight_path(request.model_path));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        if (request.config_id.has_value()) {
            throw std::runtime_error("Smart Turn does not expose selectable config assets");
        }
        const auto weight_path = resolve_weight_path(request.model_path);
        const auto assets = resolve_smart_turn_assets(weight_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = weight_path.stem().string();
        inspection.metadata.description =
            "Smart Turn v3 (pipecat-ai) turn detection: Whisper-Tiny encoder + attention-pool classifier.";
        inspection.metadata.weight_candidates = {"smart_turn.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline}},
        };
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_smart_turn_model(request);
    }
};

}  // namespace

SmartTurnSession::SmartTurnSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SmartTurnWeights> weights)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      weight_storage_type_(smart_turn_weight_type_from_options(RuntimeSessionBase::options())),
      runtime_(std::move(weights), execution_context(), weight_storage_type_) {
    if (task_.task != runtime::VoiceTaskKind::Vad) {
        throw std::runtime_error("Smart Turn only supports VoiceTaskKind::Vad");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Smart Turn only supports offline mode");
    }
}

std::string SmartTurnSession::family() const {
    return "smart_turn";
}

runtime::VoiceTaskKind SmartTurnSession::task_kind() const {
    return task_.task;
}

runtime::RunMode SmartTurnSession::run_mode() const {
    return task_.mode;
}

void SmartTurnSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("Smart Turn prepare() requires an audio contract");
    }
    mark_prepared();
}

runtime::TaskResult SmartTurnSession::run(const runtime::TaskRequest & request) {
    require_prepared("Smart Turn run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Smart Turn run() requires audio_input");
    }
    const auto wall_start = std::chrono::steady_clock::now();
    const auto & config = runtime_.config();
    const float threshold = runtime::parse_float_option(
        request.options,
        {"smart_turn.threshold", "threshold"}).value_or(config.threshold);

    const auto inference = runtime_.infer_audio(*request.audio_input);
    const int prediction = inference.probability > threshold ? 1 : 0;

    runtime::TaskResult result;
    // Speech-segment representation of the turn decision for CLI/JSON consumers.
    runtime::SpeechSegment segment;
    const auto & audio = *request.audio_input;
    segment.span.end_sample = static_cast<int64_t>(audio.samples.size());
    segment.confidence = inference.probability;
    segment.text = prediction == 1 ? "turn_complete" : "turn_incomplete";
    result.speech_segments.push_back(std::move(segment));

    // Structured turn-detection artifact with the raw probability.
    std::ostringstream json;
    json << "{\"prediction\":" << prediction
         << ",\"probability\":" << inference.probability
         << ",\"threshold\":" << threshold
         << ",\"model\":\"smart-turn-v3\"}";
    result.artifact_output = runtime::make_voice_artifact(
        runtime::ArtifactKind::VadState,
        "smart_turn",
        runtime::bytes_from_string(json.str()));

    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

SmartTurnLoadedModel::SmartTurnLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const SmartTurnWeights> weights)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      weights_(std::move(weights)) {}

const runtime::ModelMetadata & SmartTurnLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & SmartTurnLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> SmartTurnLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<SmartTurnSession>(task, options, weights_);
}

std::unique_ptr<SmartTurnLoadedModel> load_smart_turn_model(const runtime::ModelLoadRequest & request) {
    if (request.config_id.has_value()) {
        throw std::runtime_error("Smart Turn does not expose selectable config assets");
    }
    const auto weights_candidates = discover_weight_assets(request);
    const auto * selected_weight = runtime::select_named_asset(weights_candidates, request.weight_id, "weight");
    const auto weight_path = selected_weight != nullptr ? selected_weight->path : resolve_weight_path(request.model_path);
    const auto assets = resolve_smart_turn_assets(weight_path);
    runtime::ModelMetadata metadata;
    metadata.family = "smart_turn";
    metadata.variant = weight_path.stem().string();
    metadata.description =
        "Smart Turn v3 (pipecat-ai) turn detection: Whisper-Tiny encoder + attention-pool classifier.";
    metadata.weight_candidates = {"smart_turn.safetensors"};
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline}},
    };
    return std::make_unique<SmartTurnLoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        load_smart_turn_weights_cached(assets.checkpoint_path));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_smart_turn_loader() {
    return std::make_shared<SmartTurnLoader>();
}

}  // namespace engine::models::smart_turn
