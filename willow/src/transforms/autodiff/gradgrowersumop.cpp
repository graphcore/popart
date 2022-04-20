
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <memory>
#include <transforms/autodiff/gradgrowersumop.hpp>
#include <utility>
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/op/sum.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

#include "popart/names.hpp"
#include "popart/operators.hpp"
#include "popart/tensordebuginfo.hpp"
#include "transforms/autodiff/autodiffhelper.hpp"

namespace popart {
class AutodiffIrInterface;

GradGrowerSumOp::GradGrowerSumOp(AutodiffIrInterface &dep)
    : GradGrowerSumOpInterface(), AutodiffHelper(dep) {}

Op *GradGrowerSumOp::growGradSumOp(Graph &bwdGraph,
                                   Tensor *target,
                                   const std::vector<Tensor *> &toSum,
                                   AliasModel &bwdGraphAliases) {
  TensorId gradientId =
      fwdIdToBwdGradId(target->getGraph(), bwdGraph, target->id);

  // TODO: T36603 Growing the grad sum with a fixed version may result
  // in suboptimal outlining (it's included as an outline attribute).
  auto uniqOp = std::make_unique<SumOp>(
      Onnx::Operators::Sum_8,
      Op::Settings{bwdGraph, getGradSumOpNamePrefix() + "_" + gradientId});
  auto opId = bwdGraph.moveIntoGraph(std::move(uniqOp));

  std::vector<TensorId> inputs;
  inputs.reserve(toSum.size());
  for (auto &tensor : toSum) {
    inputs.push_back(tensor->id);
  }
  std::vector<TensorId> outputs{gradientId};
  bwdGraph.connectInputs(InputVecWrapper(inputs), opId);
  bwdGraph.connectOutputs(OutputVecWrapper(outputs), opId);
  Op *op = bwdGraph.getOps()[opId].get();
  op->setup();
  op->inheritPlacementAttributes(true, bwdGraphAliases);
  return op;
}

std::string GradGrowerSumOp::getGradSumOpNamePrefix() { return "GradSum"; }

} // namespace popart
