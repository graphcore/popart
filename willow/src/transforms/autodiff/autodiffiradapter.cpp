// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/autodiffiradapter.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

namespace popart {

AutodiffIrAdapter::AutodiffIrAdapter(Ir &ir_) : ir(ir_) {}

Graph &AutodiffIrAdapter::getMainGraph() { return ir.get().getMainGraph(); }

std::vector<const Graph *> AutodiffIrAdapter::getGraphSchedule() {
  return ir.get().getGraphSchedule();
}

std::vector<const Graph *> AutodiffIrAdapter::getGraphSchedule(GraphId root) {
  return ir.get().getGraphSchedule(root);
}

bool AutodiffIrAdapter::hasGraph(const GraphId &id) const {
  return ir.get().hasGraph(id);
}

Graph &AutodiffIrAdapter::getGraph(const GraphId &id) {
  return ir.get().getGraph(id);
}

Graph &AutodiffIrAdapter::createGraph(const GraphId &id) {
  return ir.get().createGraph(id);
}

Tensors &AutodiffIrAdapter::getTensors() {
  return ir.get().getMainGraph().getTensors();
}

const SessionOptions &AutodiffIrAdapter::getSessionOptions() {
  return ir.get().getSessionOptions();
}

const Optimizer &AutodiffIrAdapter::getOptimizer() {
  return ir.get().getOptimizer();
}

TensorId AutodiffIrAdapter::getFinalLossId() {
  return ir.get().getFinalLossId();
}

OpId AutodiffIrAdapter::getFinalLossOpId() {
  return ir.get().getFinalLossOpId();
}

int AutodiffIrAdapter::getOpSetVersionFromModel(const std::string &domain) {
  return ir.get().getOpSetVersionFromModel(domain);
}

void AutodiffIrAdapter::setMainGraphPathFromLoss() {
  ir.get().setMainGraphPathFromLoss();
}

PipelineStage AutodiffIrAdapter::getFinalLossPipelineStage() {
  return ir.get().getFinalLossPipelineStage();
}

PipelineStage AutodiffIrAdapter::getMaxPipelineStage() {
  return ir.get().getMaxPipelineStage();
}

TensorId
AutodiffIrAdapter::createIntermediateTensorId(const TensorId &base_id) {
  return ir.get().createIntermediateTensorId(base_id);
}

} // namespace popart