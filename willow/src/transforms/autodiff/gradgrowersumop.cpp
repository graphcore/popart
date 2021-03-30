
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrowersumop.hpp>

#include <memory>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

GradGrowerSumOp::GradGrowerSumOp(AutodiffIrInterface &dep)
    : GradGrowerSumOpInterface(), GradGrower(dep) {}

Op *GradGrowerSumOp::growGradSumOp(Tensor *target,
                                   const std::vector<Tensor *> &toSum) {
  TensorId gradientId = getGradId(target->id);

  std::unique_ptr<popart::Op> gradSum =
      OpManager::createOp(Domain::ai_onnx,
                          "Sum",
                          dep.get().getOpSetVersionFromModel(Domain::ai_onnx),
                          dep.get().getMainGraph(),
                          getGradSumOpNamePrefix() + "_" + gradientId);

  OpId opId = dep.get().getMainGraph().moveIntoGraph(std::move(gradSum));

  std::vector<TensorId> inputs;
  inputs.reserve(toSum.size());
  for (auto &tensor : toSum) {
    inputs.push_back(tensor->id);
  }
  std::vector<TensorId> outputs{gradientId};

  dep.get().getMainGraph().connectInputs(InputVecWrapper(inputs), opId);
  dep.get().getMainGraph().connectOutputs(OutputVecWrapper(outputs), opId);
  Op *op = dep.get().getMainGraph().getOps()[opId].get();
  op->setup();
  op->inheritPlacementAttributes(true);
  return op;
}

std::string GradGrowerSumOp::getGradSumOpNamePrefix() { return "GradSum"; }

} // namespace popart
