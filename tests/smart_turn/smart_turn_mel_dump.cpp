// Diagnostic: dump the native Smart Turn log-mel features for a WAV file so
// they can be compared offline against the HuggingFace reference features.
//
// Usage: smart_turn_mel_dump <model_dir> <wav_path> <out_raw_path>

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/smart_turn/runtime.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_dir = argc > 1 ? argv[1] : "models/smart_turn";
        const std::filesystem::path wav_path = argc > 2 ? argv[2] : "";
        const std::filesystem::path out_path = argc > 3 ? argv[3] : "mel.raw";
        if (wav_path.empty()) {
            std::cerr << "usage: smart_turn_mel_dump <model_dir> <wav_path> <out_raw_path>\n";
            return 1;
        }

        const auto assets = engine::models::smart_turn::resolve_smart_turn_assets(model_dir);
        const auto weights = engine::models::smart_turn::load_smart_turn_weights_cached(assets.checkpoint_path);
        engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, 1});
        engine::models::smart_turn::SmartTurnRuntime runtime(weights, execution, engine::assets::TensorStorageType::Native);

        const auto wav = engine::audio::read_wav_f32(wav_path);
        engine::runtime::AudioBuffer audio;
        audio.sample_rate = wav.sample_rate;
        audio.channels = wav.channels;
        audio.samples = wav.samples;
        const auto mel = runtime.extract_audio_features(audio);

        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(mel.data()), static_cast<std::streamsize>(mel.size() * sizeof(float)));
        out.close();
        std::cout << "wrote " << mel.size() << " floats ("
                  << weights->config.n_mels << " x " << weights->config.mel_frames
                  << ") to " << out_path << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "smart_turn_mel_dump failed: " << ex.what() << "\n";
        return 1;
    }
}
