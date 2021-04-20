// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/opidentifier.hpp"
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/hostcopy.hpp>
#include <popart/op/init.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
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

  if (pass == 1) {
    for (auto &t : graph.getTensors().getOfType(TensorType::Stream)) {
      setupHostLoadOps(t);
    }
  }

  if (pass == 2) {
    for (auto &t : ir.getAnchorRemap().leftMap()) {
      setupHostStoreOps(ir.getTensor(t.first));
    }
  }

  logging::debug("[HostIOSetup] Done.");
  return true;
}

void HostIOSetup::setupHostLoadOps(Tensor *inTensor) const {

  auto &graph             = inTensor->getGraph();
  auto &ir                = graph.getIr();
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
  settings.schedulePriority = std::numeric_limits<double>::lowest();

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

  auto vgID = inTensor->consumers.findLowestVirtualGraphID();

  initOp->setVirtualGraphId(vgID);
  hostLoadOp->setVirtualGraphId(vgID);

  if (ir.getSessionOptions().enablePipelining) {
    auto plStage = inTensor->consumers.findLowestPipelineStage();
    initOp->setPipelineStage(plStage);
    hostLoadOp->setPipelineStage(plStage);
  }

  initOp->setup();
  hostLoadOp->setup();

  for (auto consumer : inTensor->consumers.getOps()) {
    for (auto index : consumer->input->indices(inTensor)) {
      consumer->disconnectInTensor(index);
      consumer->connectInTensor(index, loadId);
    }
  }
}

void HostIOSetup::setupHostStoreOps(Tensor *anchorTensor) const {

  auto &graph = anchorTensor->getGraph();
  auto &ir    = graph.getIr();

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
  settings.schedulePriority = std::numeric_limits<double>::lowest();

  auto hostStoreOpUp = std::make_unique<HostStoreOp>(
      Onnx::CustomOperators::HostStore, settings, streamTensorId);

  OpId hostStoreOpId = graph.moveIntoGraph(std::move(hostStoreOpUp));
  Op *hostStoreOp    = graph.getOp(hostStoreOpId);

  hostStoreOp->connectInTensor(HostStoreOp::getLocalTensorInIndex(),
                               anchorTensor->id);

  if (anchorTensor->hasProducer()) {
    auto producer = anchorTensor->getProducer();

    if (producer->hasVirtualGraphId()) {
      hostStoreOp->setVirtualGraphId(producer->getVirtualGraphId());
    }

    if (ir.getSessionOptions().enablePipelining &&
        producer->hasPipelineStage()) {
      hostStoreOp->setPipelineStage(producer->getPipelineStage());
    }
  }

  hostStoreOp->setup();
}

namespace {
// HostIOSetup 1: Inputs to HostLoad Ops
bool init1 = Transform::registerTransform(new HostIOSetup(1));
// HostIOSetup 2: Anchors to HostStore Ops
bool init2 = Transform::registerTransform(new HostIOSetup(2));
} // namespace

} // namespace popart
