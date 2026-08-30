#include "engine/models/smart_turn/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::smart_turn {
namespace {

using Clock = std::chrono::steady_clock;

std::shared_ptr<const SmartTurnWeights> require_weights(std::shared_ptr<const SmartTurnWeights> weights) {
    if (weights == nullptr) {
        throw std::runtime_error("Smart Turn runtime requires weights");
    }
    return weights;
}

size_t resolve_feature_threads(const core::ExecutionContext & execution_context) {
    const int configured = execution_context.config().threads;
    return configured > 0 ? static_cast<size_t>(configured) : 0U;
}

}  // namespace

struct SmartTurnRuntime::InferenceGraph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    core::TensorValue input;
    core::TensorValue output;

    ~InferenceGraph() {
        release();
    }

    void release() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        graph = nullptr;
        input = {};
        output = {};
    }
};

SmartTurnRuntime::SmartTurnRuntime(
    std::shared_ptr<const SmartTurnWeights> weights,
    core::ExecutionContext & execution_context,
    assets::TensorStorageType weight_storage_type)
    : weights_(require_weights(std::move(weights))),
      extractor_(engine::audio::WhisperLogMelConfig{
          weights_->config.sample_rate,
          weights_->config.n_fft,
          weights_->config.hop_length,
          weights_->config.n_mels,
          engine::audio::STFTFamily::Kokoro,
      }),
      execution_context_(&execution_context) {
    const auto & cfg = weights_->config;
    const int64_t d = cfg.d_model;
    encoder_config_ = modules::WhisperEmbeddingConfig{
        cfg.n_mels,
        cfg.n_audio_ctx,
        cfg.d_model,
        cfg.n_audio_head,
        cfg.n_audio_layer,
        cfg.layer_norm_eps,
    };

    auto backend_weights = std::make_shared<SmartTurnBackendWeights>();
    backend_weights->execution_context = std::make_shared<core::ExecutionContext>(execution_context.config());
    backend_weights->store = std::make_shared<core::BackendWeightStore>(
        backend_weights->execution_context->backend(),
        backend_weights->execution_context->backend_type(),
        "smart_turn.weights",
        256ull * 1024ull * 1024ull);

    auto & store = *backend_weights->store;
    auto & source = *weights_->source;
    modules::WhisperEmbeddingWeights & embedding = backend_weights->encoder;
    embedding.conv1 = {
        store.load_tensor(
            source,
            "model.encoder.conv1.weight",
            weight_storage_type,
            {d, cfg.n_mels, 3}),
        store.load_f32_tensor(source, "model.encoder.conv1.bias", {d}),
    };
    embedding.conv2 = {
        store.load_tensor(
            source,
            "model.encoder.conv2.weight",
            weight_storage_type,
            {d, d, 3}),
        store.load_f32_tensor(source, "model.encoder.conv2.bias", {d}),
    };
    embedding.positional_embedding = store.load_f32_tensor(
        source,
        "model.encoder.embed_positions.weight",
        {cfg.n_audio_ctx, d});
    embedding.layers.reserve(static_cast<size_t>(cfg.n_audio_layer));
    for (int64_t layer = 0; layer < cfg.n_audio_layer; ++layer) {
        const std::string prefix = "model.encoder.layers." + std::to_string(layer);
        modules::WhisperEncoderLayerWeights layer_weights;
        layer_weights.attention_norm = modules::binding::norm_from_source(
            store, source, prefix + ".self_attn_layer_norm", d);
        layer_weights.attention.query = modules::binding::linear_from_source(
            store, source, prefix + ".self_attn.q_proj", weight_storage_type, d, d, true);
        layer_weights.attention.key = modules::binding::linear_from_source(
            store, source, prefix + ".self_attn.k_proj", weight_storage_type, d, d, false);
        layer_weights.attention.value = modules::binding::linear_from_source(
            store, source, prefix + ".self_attn.v_proj", weight_storage_type, d, d, true);
        layer_weights.attention.out = modules::binding::linear_from_source(
            store, source, prefix + ".self_attn.out_proj", weight_storage_type, d, d, true);
        layer_weights.mlp_norm = modules::binding::norm_from_source(
            store, source, prefix + ".final_layer_norm", d);
        layer_weights.mlp.fc1_weight = store.load_tensor(
            source, prefix + ".fc1.weight", weight_storage_type, {cfg.ffn_dim, d});
        layer_weights.mlp.fc1_bias = store.load_f32_tensor(
            source, prefix + ".fc1.bias", {cfg.ffn_dim});
        layer_weights.mlp.fc2_weight = store.load_tensor(
            source, prefix + ".fc2.weight", weight_storage_type, {d, cfg.ffn_dim});
        layer_weights.mlp.fc2_bias = store.load_f32_tensor(
            source, prefix + ".fc2.bias", {d});
        embedding.layers.push_back(std::move(layer_weights));
    }
    embedding.final_norm = modules::binding::norm_from_source(
        store, source, "model.encoder.layer_norm", d);

    // Attention-pool + classifier head weights share the encoder weight store.
    backend_weights->pool_attention_0 = modules::binding::linear_from_source(
        store, source, "pool_attention.0", weight_storage_type, cfg.pool_dim, d, true);
    backend_weights->pool_attention_2 = modules::binding::linear_from_source(
        store, source, "pool_attention.2", weight_storage_type, 1, cfg.pool_dim, true);
    backend_weights->classifier_0 = modules::binding::linear_from_source(
        store, source, "classifier.0", weight_storage_type, cfg.classifier_hidden, d, true);
    backend_weights->classifier_1 = modules::binding::norm_from_source(
        store, source, "classifier.1", cfg.classifier_hidden);
    backend_weights->classifier_4 = modules::binding::linear_from_source(
        store, source, "classifier.4", weight_storage_type, cfg.classifier_mid, cfg.classifier_hidden, true);
    backend_weights->classifier_6 = modules::binding::linear_from_source(
        store, source, "classifier.6", weight_storage_type, 1, cfg.classifier_mid, true);

    store.upload();
    source.release_storage();

    backend_weights_ = std::move(backend_weights);
}

SmartTurnRuntime::~SmartTurnRuntime() = default;

std::vector<float> SmartTurnRuntime::extract_audio_features(const runtime::AudioBuffer & audio) const {
    const auto & cfg = weights_->config;
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Smart Turn requires a positive sample rate");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("Smart Turn requires a positive channel count");
    }
    const auto prepare_start = Clock::now();

    // Keep the last chunk_samples, left-pad with zeros (matches pipecat's
    // truncate_audio_to_last_n_seconds in smart-turn audio_utils.py). Audio
    // already in the target mono layout skips the conversion pass.
    std::vector<float> window(static_cast<size_t>(cfg.chunk_samples), 0.0F);
    if (audio.channels == 1 && audio.sample_rate == cfg.sample_rate) {
        const size_t count = std::min(audio.samples.size(), window.size());
        std::copy(audio.samples.end() - static_cast<ptrdiff_t>(count), audio.samples.end(),
                  window.end() - static_cast<ptrdiff_t>(count));
    } else {
        auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples,
            audio.sample_rate,
            audio.channels,
            static_cast<int>(cfg.sample_rate));
        if (!mono.empty()) {
            const size_t count = std::min(mono.size(), window.size());
            std::copy(mono.end() - static_cast<ptrdiff_t>(count), mono.end(),
                      window.end() - static_cast<ptrdiff_t>(count));
        }
    }

    // HF WhisperFeatureExtractor(do_normalize=True) zero-mean unit-variance
    // normalizes the padded waveform window before the STFT
    // (transformers zero_mean_unit_var_norm, population variance, eps 1e-7).
    double sum = 0.0;
    for (const float value : window) {
        sum += value;
    }
    const double mean = sum / static_cast<double>(window.size());
    double squared_sum = 0.0;
    for (const float value : window) {
        const double delta = value - mean;
        squared_sum += delta * delta;
    }
    const double variance = squared_sum / static_cast<double>(window.size());
    const float denom = static_cast<float>(std::sqrt(variance + 1.0e-7));
    for (float & value : window) {
        value = (value - static_cast<float>(mean)) / denom;
    }

    auto features = extractor_.compute(window, resolve_feature_threads(*execution_context_));
    if (features.mel_bins != cfg.n_mels || features.frames != cfg.mel_frames) {
        throw std::runtime_error("Smart Turn log-mel frontend returned an unexpected shape");
    }
    engine::debug::timing_log_scalar("smart_turn.frontend.audio_prepare_ms", engine::debug::elapsed_ms(prepare_start));
    return std::move(features.values);
}

SmartTurnRuntime::InferenceGraph & SmartTurnRuntime::ensure_inference_graph() {
    if (inference_graph_ != nullptr) {
        return *inference_graph_;
    }
    const auto build_start = Clock::now();
    const auto & cfg = weights_->config;

    ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize Smart Turn inference graph context");
    }
    inference_graph_ = std::make_unique<InferenceGraph>();
    inference_graph_->ctx = ctx;

    core::ModuleBuildContext build_ctx{ctx, "smart_turn", execution_context_->backend_type()};
    auto input = core::make_tensor(
        build_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, cfg.n_mels, cfg.mel_frames}));

    // Whisper-Tiny encoder (conv frontend, positional embeddings, transformer
    // layers, final LayerNorm) fused with the head into a single graph.
    auto x = modules::WhisperEmbeddingModule(encoder_config_)
                 .build(build_ctx, input, backend_weights_->encoder);

    // pool_attention: Linear -> Tanh -> Linear producing per-position logits.
    auto pool = modules::LinearModule({cfg.d_model, cfg.pool_dim, true})
                    .build(build_ctx, x, backend_weights_->pool_attention_0);
    pool = modules::TanhModule().build(build_ctx, pool);
    pool = modules::LinearModule({cfg.pool_dim, 1, true})
               .build(build_ctx, pool, backend_weights_->pool_attention_2);

    // Softmax over the n_audio_ctx positions.
    auto flat = core::reshape_tensor(
        build_ctx,
        core::ensure_backend_addressable_layout(build_ctx, pool),
        core::TensorShape::from_dims({cfg.n_audio_ctx}));
    auto weights_softmax = core::wrap_tensor(
        ggml_soft_max(build_ctx.ggml, flat.tensor),
        flat.shape,
        GGML_TYPE_F32);
    weights_softmax = core::reshape_tensor(
        build_ctx,
        weights_softmax,
        core::TensorShape::from_dims({1, cfg.n_audio_ctx, 1}));

    // Attention-weighted sum over positions: broadcast multiply then reduce.
    auto weighted = core::wrap_tensor(
        ggml_mul(build_ctx.ggml, x.tensor, weights_softmax.tensor),
        x.shape,
        GGML_TYPE_F32);
    auto transposed = modules::TransposeModule({std::array<int, core::kMaxTensorRank>{0, 2, 1}, weighted.shape.rank})
                          .build(build_ctx, weighted);
    transposed = core::ensure_backend_addressable_layout(build_ctx, transposed);
    auto summed = core::wrap_tensor(
        ggml_sum_rows(build_ctx.ggml, transposed.tensor),
        core::TensorShape::from_dims({1, cfg.d_model, 1}),
        GGML_TYPE_F32);
    auto pooled = core::reshape_tensor(
        build_ctx,
        summed,
        core::TensorShape::from_dims({1, cfg.d_model}));

    // classifier: Linear -> LayerNorm -> GELU -> Linear -> GELU -> Linear -> Sigmoid.
    auto h = modules::LinearModule({cfg.d_model, cfg.classifier_hidden, true})
                 .build(build_ctx, pooled, backend_weights_->classifier_0);
    h = modules::LayerNormModule({cfg.classifier_hidden, cfg.layer_norm_eps, true, true})
            .build(build_ctx, h, backend_weights_->classifier_1);
    h = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(build_ctx, h);
    h = modules::LinearModule({cfg.classifier_hidden, cfg.classifier_mid, true})
            .build(build_ctx, h, backend_weights_->classifier_4);
    h = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(build_ctx, h);
    h = modules::LinearModule({cfg.classifier_mid, 1, true})
            .build(build_ctx, h, backend_weights_->classifier_6);
    auto probability = modules::SigmoidModule().build(build_ctx, h);

    ggml_set_output(probability.tensor);
    inference_graph_->input = input;
    inference_graph_->output = probability;
    inference_graph_->graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(inference_graph_->graph, probability.tensor);
    inference_graph_->gallocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (inference_graph_->gallocr == nullptr ||
        !ggml_gallocr_reserve(inference_graph_->gallocr, inference_graph_->graph) ||
        !ggml_gallocr_alloc_graph(inference_graph_->gallocr, inference_graph_->graph)) {
        inference_graph_->release();
        throw std::runtime_error("failed to allocate Smart Turn inference graph tensors");
    }
    engine::debug::timing_log_scalar("smart_turn.graph.build_ms", engine::debug::elapsed_ms(build_start));
    return *inference_graph_;
}

SmartTurnInferenceResult SmartTurnRuntime::infer_features(const std::vector<float> & mel_features) {
    const auto & cfg = weights_->config;
    if (mel_features.size() != static_cast<size_t>(cfg.n_mels * cfg.mel_frames)) {
        throw std::runtime_error("Smart Turn feature tensor size mismatch");
    }

    auto & graph = ensure_inference_graph();
    const auto compute_start = Clock::now();
    core::write_tensor_f32(graph.input, mel_features);
    if (core::compute_backend_graph(execution_context_->backend(), graph.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_backend_graph_compute failed for Smart Turn");
    }
    auto output = core::read_tensor_f32(graph.output.tensor);
    engine::debug::timing_log_scalar("smart_turn.graph.compute_ms", engine::debug::elapsed_ms(compute_start));

    SmartTurnInferenceResult result;
    result.probability = output.empty() ? 0.0F : output.front();
    return result;
}

SmartTurnInferenceResult SmartTurnRuntime::infer_audio(const runtime::AudioBuffer & audio) {
    return infer_features(extract_audio_features(audio));
}

}  // namespace engine::models::smart_turn
