#include "../core/audio_task_warm_bench.h"

int main(int argc, char ** argv) {
    try {
        engine::tools::AudioTaskBenchConfig config;
        config.family = "smart_turn";
        config.default_model = "models/smart_turn";
        config.task = engine::runtime::VoiceTaskKind::Vad;
        config.output_kind = engine::tools::AudioTaskOutputKind::Vad;
        return engine::tools::run_audio_task_warm_bench(
            argc,
            argv,
            config);
    } catch (const std::exception & ex) {
        std::cerr << "smart_turn_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
