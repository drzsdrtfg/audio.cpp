#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/smart_turn/assets.h"
#include "engine/models/smart_turn/runtime.h"

#include <memory>

namespace engine::models::smart_turn {

class SmartTurnSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    SmartTurnSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SmartTurnWeights> weights);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    assets::TensorStorageType weight_storage_type_;
    SmartTurnRuntime runtime_;
};

class SmartTurnLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    SmartTurnLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const SmartTurnWeights> weights);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const SmartTurnWeights> weights_;
};

std::unique_ptr<SmartTurnLoadedModel> load_smart_turn_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_smart_turn_loader();

}  // namespace engine::models::smart_turn
