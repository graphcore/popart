// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrowermaingraph.hpp>

#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>

#include <transforms/autodiff/backwardsgraphcreator.hpp>
#include <transforms/autodiff/gradnongradpair.hpp>
#include <transforms/autodiff/opgradregistry.hpp>
#include <transforms/autodiff/recomputestitcher.hpp>
#include <transforms/autodiff/tensorgradregistry.hpp>

namespace popart {

GradGrowerMainGraph::GradGrowerMainGraph(
    AutodiffIrInterface &dep,
    std::unique_ptr<GradGrowerOpInterface> gradOpGrower_,
    std::unique_ptr<GradGrowerLossInterface> gradLossGrower_,
    std::unique_ptr<GradGrowerSumOpInterface> gradSumOpGrower_,
    std::unique_ptr<GradGrowerGraphInterface> gradGraphGrower_)
    : GradGrowerMainGraphInterface(), AutodiffHelper(dep),
      gradOpGrower(std::move(gradOpGrower_)),
      gradLossGrower(std::move(gradLossGrower_)),
      gradSumOpGrower(std::move(gradSumOpGrower_)),
      gradGraphGrower(std::move(gradGraphGrower_)) {}

void GradGrowerMainGraph::growGradMainGraph() {

  auto &mainGraph = dep.get().getMainGraph();

  AliasModel mainGraphAliasModel;
  AliasModelGrower aliasModelGrower{mainGraphAliasModel};
  aliasModelGrower.growFullGraph(dep.get().getMainGraph(),
                                 DataDependenciesOnly::Yes);

  // We need to grow called graphs before we grow the main graph.
  FwdGraphToBwdGraphInfo calledGraphGradInfo;

  for (auto g : mainGraph.getCalledGraphs()) {

    if (calledGraphGradInfo.find(g->id) == calledGraphGradInfo.end()) {
      calledGraphGradInfo = gradGraphGrower->growBackwardsGraph(
          g->id,
          // Assume we provide all fwd outputs as gradients.
          g->getOutputIds(),
          // Require no fwd inputs as gradients as some won't be possible.
          nonstd::optional<std::vector<TensorId>>(),
          calledGraphGradInfo,
          dep.get().getSessionOptions().autodiffSettings.stitchStrategy);
    }
  }

  // definition: edge-gradient. What is output by a grad-op,
  // and which will be summed with other edge-gradients to create
  // a gradient. It is possible that an edge-gradient has the same
  // value as a gradient, if a tensor has only 1 consumer.

  // design decision w.r.t. lambda functions in this function:
  // see-sawing between lambda functions (see two following here)
  // and member functions. In general I don't like lambda functions,
  // their return types are not easily visible and capturing parameters
  // is tedious. However, I also don't like having class variables
  // which are only used in one bit of functionality, because it becomes
  // unclear whether they should be maintained in a valid state throughout
  // the objects life. In this case, I think the second is worse, so
  // going for the lambda solution.

  TensorGradRegistry tensor_grad_registry{dep};
  OpGradRegistry op_grad_registry{dep};

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
    auto gradOps = gradOpGrower->growGradOps(nonGradOp, calledGraphGradInfo);
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
    Tensor *nonGrad = dep.get().getTensors().get(nongrad_egrads.first);

    // Grow grad sum.
    const std::vector<Tensor *> &egrads = nongrad_egrads.second;
    // nongrad required below, as the name of the output of the
    // created op (sumOp) will be based off of it. Also, we
    // register the link between sumOp's output and nongrad
    Op *sumOp =
        gradSumOpGrower->growGradSumOp(nonGrad, egrads, mainGraphAliasModel);
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
    auto nonGrad = dep.get().getMainGraph().getTensors().get(nonGradId);
    if (nonGrad->hasProducer()) {
      Op *producer = nonGrad->getProducer();
      op_grad_registry.fail(producer);

      logging::transform::trace("[Autodiff] Unable to create gradient sum "
                                "for '{}' (won't be able to grow grad ops for "
                                "producer '{}')",
                                nonGradId,
                                producer->str());
    } else {
      logging::transform::trace("[Autodiff] Unable to create gradient sum "
                                "for '{}'",
                                nonGradId);
    }
  };

  // Link up loss / loss scaling ops.
  Op *nonConstLossScaleOp = gradLossGrower->growLossGradients();

  // Add loss op gradients.
  auto finalLossOpFound =
      dep.get().getMainGraph().getOps().find(dep.get().getFinalLossOpId());
  if (finalLossOpFound != dep.get().getMainGraph().getOps().end()) {
    std::vector<GradNonGradPair> pairs;
    auto finalLossOp =
        dep.get().getMainGraph().getOp(dep.get().getFinalLossOpId());
    auto gradOps = gradOpGrower->growGradOps(finalLossOp, calledGraphGradInfo);
    logging::transform::trace("[Autodiff] Created {} gradient ops for '{}'",
                              gradOps.size(),
                              finalLossOp->str());
    processGradOps(gradOps, finalLossOp);
  } else {
    throw error("Call to growLossGradients, but finalLossOpId not found");
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

  if (nonConstLossScaleOp) {
    // Only now inherit attributes for the non-const loss scaling op, if there
    // was one. The reason we do it here is because the inherit function relies
    // on the op having input or output tensors linked to it to inherit the
    // attributes from, but at the time growLossGradients is called this op's
    // outputs have yet to be grown.
    nonConstLossScaleOp->inheritPlacementAttributes(true, mainGraphAliasModel);
  }

  dep.get().setMainGraphPathFromLoss();
}

} // namespace popart
