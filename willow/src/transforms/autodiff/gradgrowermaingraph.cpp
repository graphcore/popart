// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/bwdgraphinfo.hpp"
#include "transforms/autodiff/backwardsgraphcreator.hpp"
#include <transforms/autodiff/gradgrowermaingraph.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>

#include <transforms/autodiff/gradnongradpair.hpp>
#include <transforms/autodiff/opgradregistry.hpp>
#include <transforms/autodiff/tensorgradregistry.hpp>

namespace popart {

GradGrowerMainGraph::GradGrowerMainGraph(
    AutodiffIrInterface &dep,
    std::unique_ptr<GradGrowerOpInterface> gradOpGrower_,
    std::unique_ptr<GradGrowerLossInterface> gradLossGrower_,
    std::unique_ptr<GradGrowerSumOpInterface> gradSumOpGrower_)
    : GradGrowerMainGraphInterface(), GradGrower(dep),
      gradOpGrower(std::move(gradOpGrower_)),
      gradLossGrower(std::move(gradLossGrower_)),
      gradSumOpGrower(std::move(gradSumOpGrower_)) {}

void GradGrowerMainGraph::growGradMainGraph() {

  // We need to grow called graphs before we grow the main graph.
  BackwardsGraphCreator bwdGraphCreator{dep};
  FwdGraphToBwdGraphInfo calledGraphGradInfo;

  // Reverse order because parents need called graph grad info for children.
  auto graphSched = dep.get().getGraphSchedule();
  for (auto fwdGraphIt = graphSched.rbegin(); fwdGraphIt != graphSched.rend();
       ++fwdGraphIt) {

    // We need a non-const reference to the fwdGraph.
    auto &fwdGraph = dep.get().getGraph((*fwdGraphIt)->id);
    if (fwdGraph.id != dep.get().getMainGraph().id) {
      GraphId bwdGraphId = logging::format("{}_bwd", fwdGraph.id);

      // Create the bwdGraph.
      auto bwdGraphGradInfo = bwdGraphCreator.createBackwardsGraph(
          fwdGraph, bwdGraphId, calledGraphGradInfo);

      // Store the result info for parents graphs.
      calledGraphGradInfo.insert({fwdGraph.id, bwdGraphGradInfo});
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

  TensorGradRegistry tensor_grad_registry;
  OpGradRegistry op_grad_registry;

  tensor_grad_registry.initialize(dep);
  op_grad_registry.initialize(dep);

  // signal that a grad-op has created edge-gradients
  auto registerOpGrads = [&tensor_grad_registry](Op *gradOp, Op *nonGradOp) {
    for (auto &index_tensor : gradOp->output->tensorMap()) {
      int opOutInd     = index_tensor.first;
      Tensor *partGrad = index_tensor.second;
      // what input index of nonGradOp does the
      // edge-gradient correspond to?
      int nonGradInInd      = gradOp->getNonGradInIndex(opOutInd);
      Tensor *nonGradTensor = nonGradOp->input->tensor(nonGradInInd);
      tensor_grad_registry.insert(nonGradTensor, partGrad);
    }
  };

  // register an op that doesn't create any grad ops
  std::function<void(Op *)> registerOpWithoutGrads;
  registerOpWithoutGrads = [&tensor_grad_registry,
                            &registerOpWithoutGrads](Op *nonGradOp) {
    for (auto &index_tensor : nonGradOp->input->tensorMap()) {
      auto input = index_tensor.second;
      tensor_grad_registry.decrementNumberExpectedEdges(input);

      if (tensor_grad_registry.getNumberExpectedEdges(input) == 0 &&
          input->hasProducer()) {
        registerOpWithoutGrads(input->getProducer());
      }
    }
  };

  // communicate that a new gradient tensor
  // (which is a sum along edges) is ready
  auto registerTensorGrad = [this, &op_grad_registry](Tensor *sum) {
    Tensor *nonGrad = dep.get().getTensors().get(getNonGradId(sum->id));
    if (nonGrad->hasProducer()) {
      Op *producer = nonGrad->getProducer();
      // the index at which nonGrad was produced
      int index = producer->output->indices(nonGrad).at(0);
      op_grad_registry.insert(producer, index);
    }
  };

  // Link up loss / loss scaling ops.
  Op *nonConstLossScaleOp = gradLossGrower->growLossGradients();

  // grad-ops which have created edge-gradients, but the
  // edge-gradients haven't signalled their existance.
  // initialised as the gradients of the loss
  std::vector<GradNonGradPair> opsToRegister;

  // Add loss op gradients.
  auto finalLossOpFound =
      dep.get().getMainGraph().getOps().find(dep.get().getFinalLossOpId());
  if (finalLossOpFound != dep.get().getMainGraph().getOps().end()) {
    std::vector<GradNonGradPair> pairs;
    auto finalLossOp =
        dep.get().getMainGraph().getOp(dep.get().getFinalLossOpId());
    for (Op *gradOp :
         gradOpGrower->growGradOps(finalLossOp, calledGraphGradInfo)) {
      opsToRegister.push_back({gradOp, finalLossOp});
    }
  } else {
    throw error("Call to growLossGradients, but finalLossOpId not found");
  }

  while (!opsToRegister.empty() || !tensor_grad_registry.complete.empty()) {

    if (!opsToRegister.empty()) {
      auto &toRegister = opsToRegister.back();
      registerOpGrads(toRegister.grad, toRegister.nongrad);
      opsToRegister.resize(opsToRegister.size() - 1);
    }

    for (auto &nongrad_egrads : tensor_grad_registry.popComplete()) {
      Tensor *nongrad = dep.get().getTensors().get(nongrad_egrads.first);

      const std::vector<Tensor *> &egrads = nongrad_egrads.second;
      // nongrad required below, as the name of the output of the
      // created op (sumOp) will be based off of it. Also, we
      // register the link between sumOp's output and nongrad
      Op *sumOp       = gradSumOpGrower->growGradSumOp(nongrad, egrads);
      sumOp->fromLoss = PathFromLoss::Yes;

      switch (nongrad->tensorType()) {
      // if sumOp creates the gradient of an activation tensor,
      case TensorType::ActGrad: {
        registerTensorGrad(sumOp->output->tensor(0));
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
                    nongrad->tensor_type() + " tensor, " + nongrad->str());

      default:
        throw error("only handling ActGrad and Variable for now");
      }
    }

    for (Op *op : op_grad_registry.popComplete()) {
      auto gradOps = gradOpGrower->growGradOps(op, calledGraphGradInfo);
      if (gradOps.size() == 0) {
        registerOpWithoutGrads(op);
      } else {
        for (auto &gradOp : gradOps) {
          opsToRegister.push_back({gradOp, op});
        }
      }
    }
  }

  if (nonConstLossScaleOp) {
    // Only now inherit attributes for the non-const loss scaling op, if there
    // was one. The reason we do it here is because the inherit function relies
    // on the op having input or output tensors linked to it to inherit the
    // attributes from, but at the time growLossGradients is called this op's
    // outputs have yet to be grown.
    nonConstLossScaleOp->inheritPlacementAttributes(true);
  }

  dep.get().setMainGraphPathFromLoss();
}

} // namespace popart
