// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/autodiffiradapter.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

namespace popart {

AutodiffIrAdapter::AutodiffIrAdapter(Ir &ir_) : ir(ir_) {}

Graph &AutodiffIrAdapter::getMainGraph() { return ir.get().getMainGraph(); }

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

} // namespace popart