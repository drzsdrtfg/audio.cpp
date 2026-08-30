// Smart Turn parity test: runs real fixture WAVs through the native C++
// runtime (loader -> session -> run) and compares the completion probability
// against reference values produced by the official ONNX model with the
// HuggingFace Whisper feature extractor (see models/smart-turn tooling).
//
// Usage: smart_turn_parity <model_dir> <fixtures_dir>

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/smart_turn/session.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct FixtureExpectation {
    std::string name;
    int label = 0;
    float probability = 0.0F;
    int prediction = 0;
};

std::vector<FixtureExpectation> load_expectations(const std::filesystem::path & fixtures_dir) {
    const auto root = engine::io::json::parse_file(fixtures_dir / "expected.json");
    std::vector<FixtureExpectation> out;
    for (const auto & entry : root.require("fixtures").as_array()) {
        FixtureExpectation expectation;
        expectation.name = entry.require("name").as_string();
        expectation.label = static_cast<int>(entry.require("label").as_i64());
        expectation.probability = static_cast<float>(entry.require("hf_probability").as_number());
        expectation.prediction = static_cast<int>(entry.require("prediction").as_i64());
        out.push_back(std::move(expectation));
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_dir = argc > 1 ? argv[1] : "models/smart_turn";
        const std::filesystem::path fixtures_dir = argc > 2 ? argv[2] : "models/smart-turn/fixtures";

        const auto expectations = load_expectations(fixtures_dir);
        if (expectations.empty()) {
            std::cerr << "no fixtures found in " << fixtures_dir << "\n";
            return 1;
        }

        engine::runtime::ModelLoadRequest request;
        request.model_path = model_dir;
        auto loader = engine::models::smart_turn::make_smart_turn_loader();
        if (!loader->can_load(request)) {
            std::cerr << "smart_turn loader cannot load " << model_dir << "\n";
            return 1;
        }
        auto model = loader->load(request);
        engine::runtime::TaskSpec task;
        task.task = engine::runtime::VoiceTaskKind::Vad;
        task.mode = engine::runtime::RunMode::Offline;
        auto session = model->create_task_session(task, engine::runtime::SessionOptions{});

        int failures = 0;
        float worst_diff = 0.0F;
        constexpr float kProbabilityTolerance = 1.0e-4F;
        for (const auto & expectation : expectations) {
            const auto wav = engine::audio::read_wav_f32(fixtures_dir / (expectation.name + ".wav"));
            engine::runtime::AudioBuffer audio;
            audio.sample_rate = wav.sample_rate;
            audio.channels = wav.channels;
            audio.samples = wav.samples;

            session->prepare(engine::runtime::build_preparation_request(audio));
            auto * offline_session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
            if (offline_session == nullptr) {
                throw std::runtime_error("smart_turn session does not implement the offline task interface");
            }
            engine::runtime::TaskRequest task_request;
            task_request.audio_input = audio;
            const auto result = offline_session->run(task_request);

            if (!result.artifact_output.has_value()) {
                std::cerr << "FAIL " << expectation.name << ": missing turn artifact\n";
                ++failures;
                continue;
            }
            const std::string artifact_text(
                reinterpret_cast<const char *>(result.artifact_output->payload.data()),
                result.artifact_output->payload.size());
            if (result.speech_segments.size() != 1) {
                std::cerr << "FAIL " << expectation.name << ": expected one speech segment\n";
                ++failures;
                continue;
            }
            const float probability = result.speech_segments.front().confidence;
            const float diff = std::fabs(probability - expectation.probability);
            worst_diff = std::max(worst_diff, diff);
            const int prediction = probability > 0.5F ? 1 : 0;
            const bool ok = diff < kProbabilityTolerance;
            if (!ok) {
                ++failures;
            }
            std::cout << (ok ? "  [OK ] " : "  [FAIL] ") << expectation.name
                      << ": reference=" << expectation.probability
                      << " native=" << probability
                      << " |d|=" << diff
                      << " pred=" << prediction << "/" << expectation.label
                      << " artifact=" << artifact_text << "\n";
        }

        std::cout << "worst |d| = " << worst_diff << "\n";
        if (failures != 0) {
            std::cout << "smart_turn_parity FAILED (" << failures << " failures)\n";
            return 1;
        }
        std::cout << "smart_turn_parity passed (" << expectations.size() << " fixtures)\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "smart_turn_parity failed: " << ex.what() << "\n";
        return 1;
    }
}
