// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_GRADGROWER_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_GRADGROWER_HPP

#include <functional>
#include <utility>
#include <vector>

#include "popart/bwdgraphinfo.hpp"

namespace popart {
class AliasModel;
class GradGrowerOpInterface;
class GradGrowerSumOpInterface;
class Graph;
class Op;
class Tensor;

using GradOpsOfNonGradOp = std::pair<std::vector<Op *>, Op *>;
using GradNonGradTensor  = std::pair<Tensor *, Tensor *>;

class GradGrower {
public:
  GradGrower(Graph &fwdGraph);
  // All default ctor, copy, move, dtor.

  /**
   * \brief Grow the backward pass of `fwdGraph` into `bwdGraph`.
   *
   * \param bwdGraph The graph to grow the backwards pass (grad ops) into.
   * \param gradTensorsProvided Pairs of non-grad tensors in the #fwdGraph, and
   *     their corresponding grad tensors in the `bwdGraph`, which have already
   *     been created, that will seed this autodiff run. These will likely
   *     correspond to the provided gradient tensor inputs of the `bwdGraph`.
   * \param gradOpsProvided Pairs of non-grad ops in the #fwdGraph, and their
   *     corresponding grad ops in the `bwdGraph`, which have already been
   *     created  and setup, that will seed this autodiff run. This happens when
   *     the `bwdGraph` does not take a provided gradient tensor as input, but
   *     instead internally has grad ops that produce that tensor. The likely
   *     only case of this is when GradGrowerMainGraph initialises the backwards
   *     pass by growing the grad ops of the final loss.
   * \param gradOpGrower Injected `GradGrowerOpInterface` that will be used for
   *     growing grad ops on non-grad ops.
   * \param gradSumOpGrower Injected `GradGrowerSumOpInterface` that will be
   *     used for growing gradsum ops.
   * \param calledGraphGradInfo A `FwdGraphToBwdGraphInfo` that contains entries
   *     for every Graph that `fwdGraph` calls, else this is an error. It must
   *     not already contain an instance for `fwdGraph`.
   * \param bwdGraphAliasModel An AliasModel already grown for the `bwdGraph`.
   */
  void growGrads(Graph &bwdGraph,
                 const std::vector<GradNonGradTensor> &gradTensorsProvided,
                 const std::vector<GradOpsOfNonGradOp> &gradOpsProvided,
                 GradGrowerOpInterface &gradOpGrower,
                 GradGrowerSumOpInterface &gradSumOpGrower,
                 FwdGraphToBwdGraphInfo &calledGraphGradInfo,
                 AliasModel &bwdGraphAliasModel);

private:
  // The forward graph for which we will grow grads.
  std::reference_wrapper<Graph> fwdGraph;
};

} // namespace popart

#endif // GUARD_NEURALNET_TRANSFORMS_AUTODIFF_GRADGROWER_HPP
