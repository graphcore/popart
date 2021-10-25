// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/softmax.hpp>
#include <popart/optimizer.hpp>
#include <popart/transforms/ensurefp32lossscale.hpp>
#include <popart/util.hpp>

namespace popart {

bool EnsureFp32LossScale::isMixedPrecisionLossGradOp(Op *op) const {
  // All NllLoss-grad-like operations
  if (op->isConvertibleTo<NllGradOp>()) {
    return true;
  } else if (op->isConvertibleTo<SoftmaxGradDirectOp>()) {
    return true;
  } else if (op->isConvertibleTo<NlllWithSoftmaxGradDirectOp>()) {
    return true;
  }

  // L1Grad Op
  else if (op->isConvertibleTo<L1GradOp>()) {
    return true;
  }

  return false;
}

Tensor *EnsureFp32LossScale::getLossScaleInputTensor(Op *op) const {
  // All NllLoss-grad-like operations
  if (op->isConvertibleTo<NllGradOp>()) {
    return op->inTensor(NllGradOp::getGradInIndex());
  } else if (op->isConvertibleTo<SoftmaxGradDirectOp>()) {
    return op->inTensor(SoftmaxGradDirectOp::getGradProbsInIndex());
  } else if (op->isConvertibleTo<NlllWithSoftmaxGradDirectOp>()) {
    return op->inTensor(NlllWithSoftmaxGradDirectOp::getGradProbsInIndex());
  }

  // L1Grad Op
  else if (op->isConvertibleTo<L1GradOp>()) {
    return op->inTensor(L1GradOp::getGradInIndex());
  }

  throw internal_error("EnsureFp32LossScale Pattern: unexpected op type.");
}

bool EnsureFp32LossScale::isPassThroughOp(Op *op) const {
  // Single input op
  if (op->input->n() == 1) {

    // All outputs must have same type as input
    DataType inputType = op->input->tensors().front()->info.dataType();
    for (Tensor *output : op->output->tensors()) {
      if (output->info.dataType() != inputType) {
        return false;
      }
    }
    return true;
  }

  return false;
}

FromLossScaleTraversalOps
EnsureFp32LossScale::traverseFromLossScaleTensor(const Graph &graph) const {
  Tensor *lossScaleTensor = getLossScaleTensor(graph);

  std::vector<Op *> passThroughOps, mplgoCandidates;

  auto visitor = [this, &passThroughOps, &mplgoCandidates](Tensor *t) {
    if (!(t->hasProducer())) {
      return true;
    }

    Op *producer = t->getProducer();

    // Pass-through ops. Continue traversal
    if (isPassThroughOp(producer)) {
      passThroughOps.push_back(producer);
      return true;
    }

    // MPLGO candidates. Terminate traversal here
    else {
      if (std::find(mplgoCandidates.begin(), mplgoCandidates.end(), producer) ==
          mplgoCandidates.end()) {
        mplgoCandidates.push_back(producer);
      }
      return false;
    }
  };

  graphutils::traverse(
      {lossScaleTensor},
      visitor,
      [](Op *op, Tensor *tq, Tensor *tn) { return true; }, // no filter
      graphutils::TraversalType::BreadthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Forward);

  return {passThroughOps, mplgoCandidates};
}

bool EnsureFp32LossScale::shouldApply(const Graph &graph) const {
  // Only relevant if we are training
  if (!graph.getIr().canTrain()) {
    return false;
  }

  if (!hasSingleConnectedLossScaleTensor(graph)) {
    return false;
  }
  Tensor *lossScaleTensor = getLossScaleTensor(graph);

  // Requires loss scale tensor is fp16
  if (lossScaleTensor->info.dataType() != DataType::FLOAT16) {
    return false;
  }

  return true;
}

bool EnsureFp32LossScale::canApply(const Graph &graph) const {
  auto traversalResults             = traverseFromLossScaleTensor(graph);
  std::vector<Op *> passThroughOps  = traversalResults.first;
  std::vector<Op *> mplgoCandidates = traversalResults.second;

  // All candidates must be able to handle fp16 input gradients, and
  // fp32 activations
  for (Op *candidate : mplgoCandidates) {
    if (!isMixedPrecisionLossGradOp(candidate)) {
      return false;
    }

    // The loss scale gradient input must be a scalar tensor
    if (getLossScaleInputTensor(candidate)->info.nelms() != 1) {
      return false;
    }

    // All MPLGOs must produce fp16 gradients
    for (Tensor *output : candidate->output->tensors()) {
      if (output->info.dataType() != DataType::FLOAT16) {
        return false;
      }
    }
  }

  return true;
}

bool EnsureFp32LossScale::apply(Graph &graph) const {
  if (!shouldApply(graph)) {
    return true;
  }

  if (!canApply(graph)) {
    throw error(
        "EnsureFp32LossScale: Unable to apply the transform on graph {}",
        graph.id);
  } else {
    logging::debug(
        "EnsureFp32LossScale: Conditions met to apply the transform.");
  }

  auto traversalResults             = traverseFromLossScaleTensor(graph);
  std::vector<Op *> passThroughOps  = traversalResults.first;
  std::vector<Op *> mplgoCandidates = traversalResults.second;

  // Set loss scale data type to float
  Tensor *lossScaleTensor = getLossScaleTensor(graph);
  lossScaleTensor->info.set(DataType::FLOAT);

  // Reset the tensor data with float data
  if (lossScaleTensor->hasTensorData()) {
    float lossScale    = graph.getIr().getOptimizer().getFinalLossScalingVal();
    auto convertedData = convertFloatToDataType(DataType::FLOAT, lossScale);
    lossScaleTensor->tensorData()->resetDataWithNonMatchingSize(
        lossScaleTensor->info, convertedData);
  }

  // Re-call Op setup functions to propagate the loss scale data type change
  // to the MPLGO inputs
  auto tensorTypeCheck =
      [](Op *op, Tensor *t, DataType dataType, std::string direction) {
        if (t->info.dataType() != dataType) {
          std::stringstream ss;
          ss << "EnsureFp32LossScale: Pass-through Op " << op->str();
          ss << "has non-" << dataType << " " << direction << " tensor "
             << t->id;

          throw internal_error(ss.str());
        }
        return;
      };

  for (Op *passThroughOp : passThroughOps) {
    // Confirm input is fp32
    for (Tensor *input : passThroughOp->input->tensors()) {
      tensorTypeCheck(passThroughOp, input, DataType::FLOAT, "input");
    }

    // Confirm output is fp16
    for (Tensor *output : passThroughOp->output->tensors()) {
      tensorTypeCheck(passThroughOp, output, DataType::FLOAT16, "output");
    }

    passThroughOp->setup();

    // Re-call setup function and confirm output is now fp32
    for (Tensor *output : passThroughOp->output->tensors()) {
      tensorTypeCheck(passThroughOp, output, DataType::FLOAT, "output");
    }
  }

  // A sanity check. For each consumer of lossScaleTensor re-call its Op's
  // setup function, and confirm that output type is still same
  for (Op *mplgo : mplgoCandidates) {
    std::map<TensorId, DataType> oldDataTypes, newDataTypes;
    for (Tensor *output : mplgo->output->tensors()) {
      oldDataTypes.emplace(output->id, output->info.dataType());
    }
    mplgo->setup();
    for (Tensor *output : mplgo->output->tensors()) {
      newDataTypes.emplace(output->id, output->info.dataType());
    }
    if (oldDataTypes != newDataTypes) {
      throw internal_error("EnsureFp32LossScale: the transform has modified "
                           "the output data type(s) of Op {}",
                           mplgo->str());
    }
  }

  return true;
}

std::size_t EnsureFp32LossScale::id() {
  return typeid(EnsureFp32LossScale).hash_code();
}

namespace {
bool init = Transform::registerTransform(new EnsureFp32LossScale);
}

} // namespace popart
