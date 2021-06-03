
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrowersumop.hpp>

#include <memory>

#include <popart/aliases.hpp>
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/op/sum.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

GradGrowerSumOp::GradGrowerSumOp(AutodiffIrInterface &dep)
    : GradGrowerSumOpInterface(), AutodiffHelper(dep) {}

Op *GradGrowerSumOp::growGradSumOp(Tensor *target,
                                   const std::vector<Tensor *> &toSum,
                                   Aliases &mainGraphAliases) {
  TensorId gradientId = getGradId(target->id);
  auto &mainGraph     = dep.get().getMainGraph();

  // TODO: T36603 Growing the grad sum with a fixed version may result
  // in suboptimal outlining (it's included as an outline attribute).
  auto uniqOp = std::make_unique<SumOp>(
      Onnx::Operators::Sum_8,
      Op::Settings{mainGraph, getGradSumOpNamePrefix() + "_" + gradientId});
  auto opId = mainGraph.moveIntoGraph(std::move(uniqOp));

  std::vector<TensorId> inputs;
  inputs.reserve(toSum.size());
  for (auto &tensor : toSum) {
    inputs.push_back(tensor->id);
  }
  std::vector<TensorId> outputs{gradientId};
  mainGraph.connectInputs(InputVecWrapper(inputs), opId);
  mainGraph.connectOutputs(OutputVecWrapper(outputs), opId);
  Op *op = mainGraph.getOps()[opId].get();
  op->setup();
  op->inheritPlacementAttributes(true, mainGraphAliases);
  return op;
}

std::string GradGrowerSumOp::getGradSumOpNamePrefix() { return "GradSum"; }

} // namespace popart
