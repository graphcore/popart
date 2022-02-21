// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "popart/operators.hpp"
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/init.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/transforms/hostiosetup.hpp>

namespace popart {

std::size_t HostIOSetup::id(int pass) {
  return typeid(HostIOSetup).hash_code() + pass;
}

bool HostIOSetup::apply(Graph &graph) const {
  logging::debug("[HostIOSetup] Starting.");
  auto &ir = graph.getIr();
  // For each input tensor we need a init + host load combo for each in the
  // main graph

  AliasModel aliasModel;
  AliasModelGrower aliasModelGrower{aliasModel};
  aliasModelGrower.growFullGraph(graph, DataDependenciesOnly::Yes);

  if (pass == 1) {
    for (auto &t : graph.getTensors().getOfType(TensorType::Stream)) {
      // Skip tensors that are not streamed like regular input streams
      if (!t->isOptimizerTensor() && !t->isRandomSeedTensor()) {
        setupHostLoadOps(t, aliasModel);
      }
    }
  }

  if (pass == 2) {
    for (auto &t : ir.getAnchorRemap().leftMap()) {
      setupHostStoreOps(ir.getTensor(t.first), aliasModel);
    }
  }

  logging::debug("[HostIOSetup] Done.");
  return true;
}

void HostIOSetup::setupHostLoadOps(Tensor *inTensor,
                                   AliasModel &aliasModel) const {

  auto &graph             = inTensor->getGraph();
  auto &ir                = graph.getIr();
  auto &sessionOptions    = ir.getSessionOptions();
  TensorId streamTensorId = inTensor->id;

  logging::ir::debug(
      "[HostIOSetup] HostLoadOp Started for tensor {} stream tensor ID {}",
      inTensor->id,
      streamTensorId);

  Op::Settings settings(graph, "");
  if (inTensor->getGraph().id == ir.getMainGraph().id) {
    settings.executionContext = ExecutionContext::Normal;
  } else {
    settings.executionContext = ExecutionContext::Subgraph;
  }

  const auto num_phases =
      sessionOptions.virtualGraphMode == VirtualGraphMode::ExecutionPhases
          ? sessionOptions.executionPhaseSettings.phases
          : 0;

  // Only force priority if batch serialisation or phased execution is not used
  if (sessionOptions.batchSerializationSettings.factor < 2 && num_phases < 2 &&
      !sessionOptions.explicitRecomputation &&
      !sessionOptions.enableExplicitMainLoops) {
    settings.schedulePriority = std::numeric_limits<double>::lowest();
  }

  auto init = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                       inTensor->info,
                                       TensorType::ActGrad,
                                       InitType::Zero,
                                       settings);

  OpId initOpId = graph.moveIntoGraph(std::move(init));
  Op *initOp    = graph.getOps()[initOpId].get();

  TensorId initId = ir.createIntermediateTensorId(inTensor->id);
  TensorId loadId = ir.createIntermediateTensorId(inTensor->id);

  initOp->createAndConnectOutTensor(InitOp::getOutIndex(), initId);

  auto hostLoadOpUp = std::make_unique<HostLoadOp>(
      Onnx::CustomOperators::HostLoad, settings, streamTensorId);

  OpId HostLoadOpId = graph.moveIntoGraph(std::move(hostLoadOpUp));
  Op *hostLoadOp    = graph.getOp(HostLoadOpId);

  hostLoadOp->connectInTensor(HostLoadOp::getLocalTensorInIndex(), initId);

  hostLoadOp->createAndConnectOutTensor(HostLoadOp::getLocalTensorOutIndex(),
                                        loadId);

  if (inTensor->isAnchored()) {
    ir.remapAnchor(inTensor->id, loadId);
  }

  initOp->setup();
  hostLoadOp->setup();

  for (auto consumer : inTensor->consumers.getOps()) {
    for (auto index : consumer->input->indices(inTensor)) {
      consumer->disconnectInTensor(index);
      consumer->connectInTensor(index, loadId);
    }
  }

  // Order important
  hostLoadOp->inheritPlacementAttributes(false, aliasModel);
  initOp->inheritPlacementAttributes(false, aliasModel);

  auto vgid = graph.getTensor(loadId)->getVirtualGraphIdUnsafe();
  if (vgid != unusedVGraphId) {
    initOp->setVirtualGraphId(vgid);
    hostLoadOp->setVirtualGraphId(vgid);
  }

  initOp->settings.tileSet     = inTensor->inputSettings.tileSet();
  hostLoadOp->settings.tileSet = inTensor->inputSettings.tileSet();
}

void HostIOSetup::setupHostStoreOps(Tensor *anchorTensor,
                                    AliasModel &aliasModel) const {

  auto &graph          = anchorTensor->getGraph();
  auto &ir             = graph.getIr();
  auto &sessionOptions = ir.getSessionOptions();

  TensorId streamTensorId = ir.getAnchorRemap().getRight(anchorTensor->id);

  logging::ir::debug(
      "[HostIOSetup] HostStoreOp started for tensor {} stream tensor ID {}",
      anchorTensor->id,
      streamTensorId);

  Op::Settings settings(graph, "");
  if (anchorTensor->getGraph().id == ir.getMainGraph().id) {
    settings.executionContext = ExecutionContext::Normal;
  } else {
    settings.executionContext = ExecutionContext::Subgraph;
  }

  const auto num_phases =
      sessionOptions.virtualGraphMode == VirtualGraphMode::ExecutionPhases
          ? sessionOptions.executionPhaseSettings.phases
          : 0;

  // Only force priority if batch serialisation or phased execution is not used
  if (sessionOptions.batchSerializationSettings.factor < 2 && num_phases < 2 &&
      !sessionOptions.explicitRecomputation) {
    settings.schedulePriority = std::numeric_limits<double>::lowest();
  }

  auto hostStoreOpUp = std::make_unique<HostStoreOp>(
      Onnx::CustomOperators::HostStore, settings, streamTensorId);

  OpId hostStoreOpId = graph.moveIntoGraph(std::move(hostStoreOpUp));
  Op *hostStoreOp    = graph.getOp(hostStoreOpId);

  hostStoreOp->connectInTensor(HostStoreOp::getLocalTensorInIndex(),
                               anchorTensor->id);

  hostStoreOp->inheritPlacementAttributes(false, aliasModel);

  auto vgid = anchorTensor->getVirtualGraphIdUnsafe();
  if (vgid != unusedVGraphId) {
    hostStoreOp->setVirtualGraphId(vgid);
  }

  hostStoreOp->settings.tileSet =
      ir.getDataFlow().art(streamTensorId).tileSet();

  hostStoreOp->setup();
}

namespace {
// HostIOSetup 1: Inputs to HostLoad Ops
bool init1 = Transform::registerTransform(new HostIOSetup(1));
// HostIOSetup 2: Anchors to HostStore Ops
bool init2 = Transform::registerTransform(new HostIOSetup(2));
} // namespace

} // namespace popart
