// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "gradgrower.hpp"

#include <map>
#include <memory>
#include <string>
#include <transforms/autodiff/opgradregistry.hpp>
#include <transforms/autodiff/tensorgradregistry.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>

#include "popart/error.hpp"
#include "popart/graph.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"
#include "popart/vendored/optional.hpp"
#include "popart/vertex.hpp"
#include "transforms/autodiff/gradgrowerop.hpp"
#include "transforms/autodiff/gradgrowersumop.hpp"

namespace popart {
class AliasModel;

GradGrower::GradGrower(Graph &fwdGraph_) : fwdGraph{fwdGraph_} {}

void GradGrower::growGrads(
    Graph &bwdGraph,
    const std::vector<GradNonGradTensor> &gradTensorsProvided,
    const std::vector<GradOpsOfNonGradOp> &gradOpsProvided,
    GradGrowerOpInterface &gradOpGrower,
    GradGrowerSumOpInterface &gradSumOpGrower,
    FwdGraphToBwdGraphInfo &calledGraphGradInfo,
    AliasModel &bwdGraphAliasModel) {

  TensorGradRegistry tensor_grad_registry{fwdGraph};
  OpGradRegistry op_grad_registry{fwdGraph};

  tensor_grad_registry.initialize();
  op_grad_registry.initialize();

  auto processGradOps = [&](const std::vector<Op *> &gradOps, Op *nonGradOp) {
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
  };

  // signal that a grad-op has created edge-gradients
  auto growGradOps = [&](Op *nonGradOp) {
    auto gradOps =
        gradOpGrower.growGradOps(bwdGraph, nonGradOp, calledGraphGradInfo);
    logging::transform::trace("[Autodiff] Created {} gradient ops for '{}'",
                              gradOps.size(),
                              nonGradOp->str());
    // Process actual grad ops.
    processGradOps(gradOps, nonGradOp);
  };

  // communicate an op will have no grad ops created.
  auto failGradOps = [&](Op *nonGradOp) {
    // Log a meaningful error.
    std::vector<TensorId> unavailIds;
    for (auto &outputTensor : nonGradOp->output->tensorMap()) {
      if (tensor_grad_registry.getNumberExpectedEdges(outputTensor.second) ==
          0) {
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
  };

  // Process a gradient tensor that is ready to grow.
  auto growGradSum = [&](TensorGradRegistry::TMap::value_type nongrad_egrads) {
    Tensor *nonGrad = fwdGraph.get().getTensors().get(nongrad_egrads.first);

    // Grow grad sum.
    const std::vector<Tensor *> &egrads = nongrad_egrads.second;
    // nongrad required below, as the name of the output of the
    // created op (sumOp) will be based off of it. Also, we
    // register the link between sumOp's output and nongrad
    Op *sumOp = gradSumOpGrower.growGradSumOp(
        bwdGraph, nonGrad, egrads, bwdGraphAliasModel);
    // NOTE: It is not valid to set this attribute on vertices in a subgraph,
    // however we store the information transiently here: after GradGrower runs,
    // the caller is responsible for unsetting this attribute on all vertices
    // not in the main graph.
    sumOp->fromLoss = PathFromLoss::Yes;

    logging::transform::trace("[Autodiff] Created gradient sum for '{}'",
                              nonGrad->id);

    switch (nonGrad->tensorType()) {
    // if sumOp creates the gradient of an activation tensor,
    case TensorType::ActGrad: {

      Tensor *sum = sumOp->output->tensor(0);
      // communicate that a new gradient tensor
      // (which is a sum along edges) is ready
      Tensor *nonGrad = fwdGraph.get().getTensors().get(
          bwdGradIdToFwdId(fwdGraph, bwdGraph, sum->id));
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
      throw error("can't currently register gradient of " +
                  nonGrad->tensor_type() + " tensor, " + nonGrad->str());

    default:
      throw error("only handling ActGrad and Variable for now");
    }
  };

  // communicate that a gradient will never be available
  auto failGradSum = [this, &op_grad_registry](
                         TensorGradRegistry::TMap::value_type nongrad_egrads) {
    TensorId nonGradId = nongrad_egrads.first;
    // Gradient is not available and is never going to be available.
    auto nonGrad = fwdGraph.get().getTensors().get(nonGradId);
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
  };

  // Process initial `gradTensorsProvided`.
  for (auto &entry : gradTensorsProvided) {
    // auto gradTensor    = entry.first;
    auto nonGradTensor = entry.second;

    if (nonGradTensor->hasProducer()) {
      auto producer = nonGradTensor->getProducerUnsafe();
      // the index at which nonGrad was produced
      int index = producer->output->indices(nonGradTensor).at(0);
      op_grad_registry.insert(producer, index);
    }
  }

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
      growGradSum(*entry);
      logState(logging::Level::Trace);
    }

    while (auto entry = tensor_grad_registry.popFailed()) {
      didSomething = true;
      failGradSum(*entry);
      logState(logging::Level::Trace);
    }

    while (auto op = op_grad_registry.popComplete()) {
      didSomething = true;
      growGradOps(*op);
      logState(logging::Level::Trace);
    }

    while (auto op = op_grad_registry.popFailed()) {
      didSomething = true;
      failGradOps(*op);
      logState(logging::Level::Trace);
    }

    if (!didSomething) {
      break;
    }
  }

  logState(logging::Level::Debug);
}

} // namespace popart
