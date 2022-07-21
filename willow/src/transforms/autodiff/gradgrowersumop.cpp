
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <memory>
#include <transforms/autodiff/gradgrowersumop.hpp>
#include <utility>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
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

  std::map<InIndex, TensorId> inputs;
  InIndex inCount = 0;

  if (bwdGraph.hasInputId(gradientId)) {
    // In some cases, the gradient we are summing to is provided by the user
    // as an input gradient. For example:
    //
    //              a       b
    //              |       |
    // .------------|-------|----------------[A]---.
    // |            |       |                      |
    // |            |   .---'                      |
    // |            |   |                          |
    // |            V   V                          |
    // |            AddOp                          |
    // |              |                            |
    // |              |-------.---.                |
    // |              |       |   |                |
    // |              |       V   V                |
    // |              |       AddOp                |
    // |              |         |                  |
    // '--------------|---------|------------------'
    //                |         |
    //                y         z
    //
    // Here, the user might provide the gradient of graph output y as an input.
    // However, there is still a need to create a gradient sum for this gradient
    // as other paths also contribute to the gradient of the output of the first
    // AddOp. To accomodate this edge case, we rename the graph input in the
    // backwards graph and add it as an edge to the gradient sum.

    const InIndex inputIndex = bwdGraph.getInputIndex(gradientId);
    auto graphInputId = bwdGraph.getIr().createIntermediateTensorId(gradientId);
    bwdGraph.addInput(inputIndex,
                      graphInputId,
                      toSum.at(0)->info, /* use info from first summand */
                      true /* overwrite index */);

    inputs[inCount++] = graphInputId;
  }

  for (auto &tensor : toSum) {
    inputs[inCount++] = tensor->id;
  }
  DebugInfo di({"growGradSum"}, "popartbuilder");

  // TODO: T36603 Growing the grad sum with a fixed version may result
  // in suboptimal outlining (it's included as an outline attribute).
  auto sumOp = bwdGraph.createConnectedOp<SumOp>(
      inputs,
      {{0, gradientId}},
      Onnx::Operators::Sum_8,
      Op::Settings{
          bwdGraph, getGradSumOpNamePrefix() + "_" + gradientId, di.getId()});

  sumOp->inheritPlacementAttributes(true, bwdGraphAliases);
  return sumOp;
}

std::string GradGrowerSumOp::getGradSumOpNamePrefix() { return "GradSum"; }

} // namespace popart
