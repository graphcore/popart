// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "popart/aliases.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/tensorindex.hpp"

#include "popart/tensor.hpp"
#include "popart/tensornames.hpp"

#include "transforms/autodiff/gradgrower.hpp"
#include "transforms/autodiff/opgradregistry.hpp"
#include "transforms/autodiff/tensorgradregistry.hpp"

namespace popart {

GradGrower::GradGrower(std::reference_wrapper<AutodiffIrInterface> dep)
    : AutodiffHelper{std::move(dep)} {}

void GradGrower::growGrads(
    const std::vector<GradOpsOfNonGradOp> &gradOpsProvided,
    GradGrowerOpInterface &gradOpGrower,
    GradGrowerSumOpInterface &gradSumOpGrower,
    FwdGraphToBwdGraphInfo &calledGraphGradInfo,
    AliasModel &aliasModel) {

  this->tensor_grad_registry.initialize();
  this->op_grad_registry.initialize();

  // Process initial `gradOpsProvided`.
  for (auto &entry : gradOpsProvided) {
    auto &gradOps  = entry.first;
    auto nonGradOp = entry.second;
    processGradOps(gradOps, nonGradOp);
  }

  // Log the state of registry
  auto logState = [&](logging::Level level) {
    if (logging::shouldLog(logging::Module::transform, level)) {
      logging::log(logging::Module::transform,
                   level,
                   "[Autodiff] ---------------------------");
      op_grad_registry.logDump(level);
      tensor_grad_registry.logDump(level);
      logging::log(logging::Module::transform,
                   level,
                   "[Autodiff] ---------------------------");
    }
  };

  logState(logging::Level::Trace);

  while (true) {
    bool didSomething = false;

    while (auto entry = tensor_grad_registry.popComplete()) {
      didSomething = true;
      // Grow the sum ops
      growGradSum(aliasModel, gradSumOpGrower, *entry);
      logState(logging::Level::Trace);
    }

    while (auto entry = tensor_grad_registry.popFailed()) {
      didSomething = true;
      // Communicate that a gradient will never be available.
      failGradSum(*entry);
      logState(logging::Level::Trace);
    }

    while (auto op = op_grad_registry.popComplete()) {
      didSomething = true;
      // Signal that a grad-op has created edge-gradients.
      growGradOps(*op, gradOpGrower, calledGraphGradInfo);
      logState(logging::Level::Trace);
    }

    while (auto op = op_grad_registry.popFailed()) {
      didSomething = true;
      // Fail creating a grad op for the op.
      failGradOps(*op);
      logState(logging::Level::Trace);
    }

    if (!didSomething) {
      break;
    }
  }

  logState(logging::Level::Debug);
}

void GradGrower::failGradSum(
    TensorGradRegistry::TMap::value_type nongrad_egrads) {
  TensorId nonGradId = nongrad_egrads.first;
  // Gradient is not available and is never going to be available.
  auto nonGrad = dep.get().getMainGraph().getTensors().get(nonGradId);
  if (nonGrad->hasProducer()) {
    Op *producer = nonGrad->getProducer();

    logging::transform::trace("[Autodiff] Unable to create gradient sum "
                              "for '{}' (we may not be able to grow grad ops "
                              "for producer '{}')",
                              nonGradId,
                              producer->str());

    op_grad_registry.fail(producer, producer->outIndex(nonGrad));

  } else {
    logging::transform::trace("[Autodiff] Unable to create gradient sum "
                              "for '{}'",
                              nonGradId);
  }
}

void GradGrower::growGradSum(
    AliasModel &aliasModel,
    GradGrowerSumOpInterface &gradSumOpGrower,
    TensorGradRegistry::TMap::value_type nongrad_egrads) {

  Tensor *nonGrad = dep.get().getTensors().get(nongrad_egrads.first);

  // Grow grad sum.
  const std::vector<Tensor *> &egrads = nongrad_egrads.second;
  // nongrad required below, as the name of the output of the
  // created op (sumOp) will be based off of it. Also, we
  // register the link between sumOp's output and nongrad
  Op *sumOp       = gradSumOpGrower.growGradSumOp(nonGrad, egrads, aliasModel);
  sumOp->fromLoss = PathFromLoss::Yes;

  logging::transform::trace("[Autodiff] Created gradient sum for '{}'",
                            nonGrad->id);

  switch (nonGrad->tensorType()) {
  // if sumOp creates the gradient of an activation tensor,
  case TensorType::ActGrad: {

    Tensor *sum = sumOp->output->tensor(0);
    // communicate that a new gradient tensor
    // (which is a sum along edges) is ready
    Tensor *nonGrad = dep.get().getTensors().get(getNonGradId(sum->id));
    if (nonGrad->hasProducer()) {
      Op *producer = nonGrad->getProducer();
      // the index at which nonGrad was produced
      int index = producer->output->indices(nonGrad).at(0);
      op_grad_registry.insert(producer, index);
    }
    break;
  }
  case TensorType::Variable: {
    // nothing to do, variable updates
    // follows at the end of this function
    break;
  }
  case TensorType::Stream: {
    // if the user wants the gradient of the
    // input data (unusual case) maybe we won't
    // break here. Example case : generating adversarials
    break;
  }
  case TensorType::Const: {
    break;
  }
  case TensorType::Unknown:
  case TensorType::N:
  default:
    throw error(
        "[Autodiff] Failed growing gradSum for tensor '{}'. Unhandled tensor "
        "type '{}', only handling ActGrad and Variable tensors for now",
        nonGrad->id,
        nonGrad->tensor_type());
  }
}

void GradGrower::growGradOps(Op *nonGradOp,
                             GradGrowerOpInterface &gradOpGrower,
                             FwdGraphToBwdGraphInfo &calledGraphGradInfo) {
  auto gradOps = gradOpGrower.growGradOps(nonGradOp, calledGraphGradInfo);
  logging::transform::trace("[Autodiff] Created {} gradient ops for '{}'",
                            gradOps.size(),
                            nonGradOp->str());
  // Process actual grad ops.
  processGradOps(gradOps, nonGradOp);
}

void GradGrower::processGradOps(const std::vector<Op *> &gradOps,
                                Op *nonGradOp) {
  // Add the produced tensors to the registry.
  for (const auto &gradOp : gradOps) {
    for (auto &index_tensor : gradOp->output->tensorMap()) {
      int opOutInd     = index_tensor.first;
      Tensor *partGrad = index_tensor.second;
      // what input index of nonGradOp does the
      // edge-gradient correspond to?
      int nonGradInInd      = gradOp->getNonGradInIndex(opOutInd);
      Tensor *nonGradTensor = nonGradOp->input->tensor(nonGradInInd);
      tensor_grad_registry.insert(nonGradTensor, partGrad);
    }
  }

  // If there are nonGradOp inputs for which none of the gradient ops produce
  // an output then our bookkeeping of expected number of edge gradients
  // required to grow the gradient sum of those tensors will be off-by-one.
  // We correct this here. Note that it's to be expected that some ops don't
  // produce gradients for all inputs of the nonGradOp (like Where).
  for (auto &index_tensor : nonGradOp->input->tensorMap()) {
    auto &nonGradInputIndex = index_tensor.first;

    // Check if there is an gradOp output that produces the gradient for this
    // nonGradOp input.
    bool gradientOfNonGradInputIsProduced = false;
    for (auto &gradOp : gradOps) {
      for (const auto &outMap : gradOp->gradOutToNonGradIn()) {
        if (outMap.second == nonGradInputIndex) {
          gradientOfNonGradInputIsProduced = true;
        }
      }
    }

    // If the gradient for the nonGrad input is not produced, update.
    if (!gradientOfNonGradInputIsProduced) {
      auto &input = index_tensor.second;
      tensor_grad_registry.decrementNumberExpectedEdges(input);

      logging::transform::trace(
          "[Autodiff] Adjusted number of expected edge gradients for '{}' ("
          "now {} due to absence of a grad op of '{}' that outputs this "
          "tensor)",
          input->id,
          tensor_grad_registry.getNumberExpectedEdges(input),
          nonGradOp->str());
    }
  }
}

void GradGrower::failGradOps(Op *nonGradOp) {
  // Log a meaningful error.
  std::vector<TensorId> unavailIds;
  for (auto &outputTensor : nonGradOp->output->tensorMap()) {
    if (tensor_grad_registry.getNumberExpectedEdges(outputTensor.second) == 0) {
      unavailIds.push_back(outputTensor.second->id);
    }
  }
  logging::transform::trace(
      "[Autodiff] No gradient ops created for '{}' "
      "(there is no gradient sum available for: {})",
      nonGradOp->str(),
      logging::join(unavailIds.begin(), unavailIds.end(), ", "));
  // Process empty set of grad ops.
  processGradOps({}, nonGradOp);
}

} // namespace popart
