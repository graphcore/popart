// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_GRADGROWER_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_GRADGROWER_HPP

#include "popart/alias/aliasmodel.hpp"
#include "popart/op.hpp"

#include "transforms/autodiff/autodiffhelper.hpp"
#include "transforms/autodiff/autodiffirinterface.hpp"
#include "transforms/autodiff/gradgrowergraph.hpp"
#include "transforms/autodiff/gradgrowerop.hpp"
#include "transforms/autodiff/gradgrowersumop.hpp"
#include "transforms/autodiff/opgradregistry.hpp"
#include "transforms/autodiff/tensorgradregistry.hpp"

// Cant forward declare STL containers.
#include <map>
#include <vector>

namespace popart {

using GradOpsOfNonGradOp = std::pair<std::vector<Op *>, Op *>;

class GradGrower : private AutodiffHelper {
public:
  GradGrower(std::reference_wrapper<AutodiffIrInterface> dep);
  // All default ctor, copy, move, dtor.

  // GradOpsOfNonGradOp Ops must all be from same forward/backward graph.
  // AliasModel must be for the forward graph.
  // calledGraphGradInfo must not yet contain entry for the backward graph.
  // gradOpsProvided are an optional list of Ops producing the gradient tensors
  // that seed this autodiff run. If the tensors exist but have no producer, you
  // pass an empty vector.
  void growGrads(const std::vector<GradOpsOfNonGradOp> &gradOpsProvided,
                 GradGrowerOpInterface &gradOpGrower,
                 GradGrowerSumOpInterface &gradSumOpGrower,
                 FwdGraphToBwdGraphInfo &calledGraphGradInfo,
                 AliasModel &aliasModel);

private:
  /**
   * Grow the sum ops using the given gradSumOpGrower this is used for all
   * tensors in tensor_grad_registry where ALL their required gradients
   * registered, and is thus ready to have their edge gradients summed to obtain
   * the final gradient.
   *
   * \param aliasModel The aliasModel for the tensor. Used to determine aliases
   *    when creating the grad sum.
   * \param gradSumOpGrower Interface to grow the
   *    (grad) sum op with.
   * \param nongrad_egrads Map to the non-grad tensor.
   */
  void growGradSum(AliasModel &aliasModel,
                   GradGrowerSumOpInterface &gradSumOpGrower,
                   TensorGradRegistry::TMap::value_type nongrad_egrads);

  /**
   * Communicate that a gradient will never be available. This may lead to
   * failGradSum running for this tensor's producer.
   *
   * \param nongrad_egrads Map to the non-grad tensor.
   */
  void failGradSum(TensorGradRegistry::TMap::value_type nongrad_egrads);

  /**
   * Signal that a grad-op has created edge-gradients.
   *
   * \param nonGradOp Corresponding non-grad op.
   * \param gradOpGrower Interface to grow the grad op with.
   * \param calledGraphGradInfo Mapping from fwdGraph to info on the bwdGraph.
   */
  void growGradOps(Op *nonGradOp,
                   GradGrowerOpInterface &gradOpGrower,
                   FwdGraphToBwdGraphInfo &calledGraphGradInfo);

  /**
   * Process the grad ops and ensure that information in our tensor/op
   * registries are correct.
   *
   * \param gradOps The grad ops.
   * \param nonGradOp The non-grad (forward) ops.
   */
  void processGradOps(const std::vector<Op *> &gradOps, Op *nonGradOp);

  /**
   * Fail creating a grad op for the given op.
   *
   * \param nonGradOp The grad op.
   */
  void failGradOps(Op *nonGradOp);

  TensorGradRegistry tensor_grad_registry{dep};
  OpGradRegistry op_grad_registry{dep};
};

} // namespace popart

#endif // GUARD_NEURALNET_TRANSFORMS_AUTODIFF_GRADGROWER_HPP
