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

std::size_t HostIOSetup::id() { return typeid(HostIOSetup).hash_code(); }

bool HostIOSetup::apply(Graph &graph) const {
  logging::debug("[HostIOSetup] Starting.");
  auto &ir = graph.getIr();
  // For each input tensor we need a init + host load combo for each in the
  // main graph

  for (auto t : graph.getTensors().getOfType(TensorType::Stream)) {
    setupMainGraphHostLoadOps(ir.currentHostLoadId(), t, &graph);
  }

  for (auto t : ir.getDataFlow().anchors()) {
    setupMainGraphHostStoreOps(
        ir.currentHostStoreId(), ir.getTensor(t), &graph);
  }
  logging::debug("[HostIOSetup] Done.");
  return true;
}

void HostIOSetup::setupMainGraphHostLoadOps(HostStreamId hsid,
                                            Tensor *parent,
                                            Graph *graph) const {

  logging::ir::debug("[HostIOSetup] HostLoadOp Started for tensor {}, id {}",
                     parent->id,
                     hsid);

  Op::Settings settings(*graph, "");
  settings.executionContext = ExecutionContext::Normal;
  settings.schedulePriority = std::numeric_limits<double>::lowest();

  // Change stream tensor to actgrad
  parent->setTensorType(TensorType::ActGrad);

  auto init = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                       parent->info,
                                       TensorType::ActGrad,
                                       InitType::NoInit,
                                       settings);

  OpId initOpId = graph->moveIntoGraph(std::move(init));
  Op *initOp    = graph->getOps()[initOpId].get();

  TensorId initId = parent->id + "_pre_hostload";

  initOp->createAndConnectOutTensor(InitOp::getOutIndex(), initId);

  auto hostLoadOpUp = std::make_unique<HostLoadOp>(
      Onnx::CustomOperators::HostLoad, settings, hsid);

  OpId HostLoadOpId = graph->moveIntoGraph(std::move(hostLoadOpUp));
  Op *hostLoadOp    = graph->getOps()[HostLoadOpId].get();

  hostLoadOp->connectInTensor(HostLoadOp::getLocalTensorInIndex(), initId);

  hostLoadOp->connectOutTensor(HostLoadOp::getLocalTensorOutIndex(),
                               parent->id);

  auto vgID = parent->consumers.findLowestVirtualGraphID();

  initOp->setVirtualGraphId(vgID);
  hostLoadOp->setVirtualGraphId(vgID);

  if (graph->getIr().getSessionOptions().enablePipelining) {
    auto plStage = parent->consumers.findLowestPipelineStage();
    initOp->setPipelineStage(plStage);
    hostLoadOp->setPipelineStage(plStage);
  }

  initOp->setup();
  hostLoadOp->setup();
}

void HostIOSetup::setupMainGraphHostStoreOps(HostStreamId hsid,
                                             Tensor *sourceTensor,
                                             Graph *graph) const {

  logging::ir::debug("[HostIOSetup] HostStoreOp started for tensor {}, id {}",
                     sourceTensor->id,
                     hsid);

  Op::Settings settings(*graph, "");
  settings.executionContext = ExecutionContext::Normal;
  settings.schedulePriority = std::numeric_limits<double>::lowest();

  auto hostStoreOpUp = std::make_unique<HostStoreOp>(
      Onnx::CustomOperators::HostStore, settings, hsid);

  OpId HostStoreOpId = graph->moveIntoGraph(std::move(hostStoreOpUp));
  Op *hostStoreOp    = graph->getOps()[HostStoreOpId].get();

  hostStoreOp->connectInTensor(HostStoreOp::getLocalTensorInIndex(),
                               sourceTensor->id);

  auto producer = sourceTensor->getProducer();

  if (producer->hasVirtualGraphId()) {
    hostStoreOp->setVirtualGraphId(producer->getVirtualGraphId());
  }

  if (graph->getIr().getSessionOptions().enablePipelining &&
      producer->hasPipelineStage()) {
    hostStoreOp->setPipelineStage(producer->getPipelineStage());
  }

  hostStoreOp->setup();
}

namespace {
// HostIOSetup
bool init = Transform::registerTransform(new HostIOSetup());
} // namespace

} // namespace popart
