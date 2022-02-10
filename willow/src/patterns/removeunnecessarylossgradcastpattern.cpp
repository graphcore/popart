// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/softmax.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/patterns/removeunnecessarylossgradcastpattern.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/transforms/ensurefp32lossscale.hpp>

namespace popart {

bool RemoveUnnecessaryLossGradCast::matches(Op *lossOp) const {
  // Check that the Op is a grad of mixed precision
  if (!EnsureFp32LossScale::isMixedPrecisionLossGradOp(lossOp)) {
    return false;
  }

  // Check that any of prob tensor are of float 16 (or else ther will be
  // no clipping)
  auto inTensors = lossOp->input->tensors();
  if (!std::any_of(inTensors.begin(), inTensors.end(), [](Tensor *t) {
        return t->info.dataType() == DataType::FLOAT16;
      })) {
    return false;
  }

  // Check that the loss scale is of fp16 (and later if it's preceeded by
  // a cast op) The loss is connected at the getGradInIndex() in the Op
  const Tensor *lossScaleTensor =
      EnsureFp32LossScale::getLossScaleInputTensor(lossOp);
  if (lossScaleTensor->info.dataType() != DataType::FLOAT16) {
    return false;
  }

  // Check that the loss scale has a producer
  if (!lossScaleTensor->hasProducer()) {
    return false;
  }

  // Check that the loss scale is connected to a cast operator
  auto lossProducerOp = lossScaleTensor->getProducer();
  if (!lossProducerOp->isConvertibleTo<CastOp>()) {
    return false;
  }

  return true;
}

// This pattern simply removes and reconnects a tensor, and thus
// doesn't "touch" any tensors
std::vector<const Tensor *>
RemoveUnnecessaryLossGradCast::touches(Op *op) const {
  return {};
}

bool RemoveUnnecessaryLossGradCast::apply(Op *lossOp) const {
  // Get all required variables before the reconnection
  Graph &graph = lossOp->getGraph();
  Tensor *castedLossScaleTensor =
      EnsureFp32LossScale::getLossScaleInputTensor(lossOp);
  auto castOp             = castedLossScaleTensor->getProducer();
  Tensor *lossScaleTensor = castOp->input->tensor(CastOp::getInIndex());

  // Disconnect the loss scale input to the loss operator
  // Find the loss in index
  auto lossTensorMap = lossOp->input->tensorMap();
  auto IndexCastedLossScaleTensor =
      std::find_if(std::begin(lossTensorMap),
                   std::end(lossTensorMap),
                   [&](const std::pair<int, Tensor *> &pair) {
                     return pair.second == castedLossScaleTensor;
                   });
  auto lossInIndex = IndexCastedLossScaleTensor->first;
  lossOp->disconnectInTensor(lossInIndex);

  // Disconnect the loss scale tensor from the producer (the cast Op)
  castOp->disconnectAllOutputs();

  // Remove the disconnected tensor
  graph.getTensors().remove(castedLossScaleTensor->id);

  // Disconnect the input to the cast op
  castOp->disconnectAllInputs();

  // Reconnect the input to the cast op to the loss operator
  lossOp->connectInTensor(lossInIndex, lossScaleTensor->id);

  return true;
}

namespace {
// Register the pattern
static PatternCreator<RemoveUnnecessaryLossGradCast>
    removeUnnecessaryLossGradCast("RemoveUnnecessaryLossGradCast",
                                  /* enabled = */ true,
                                  /* mandatory = */ false);
} // namespace

} // namespace popart
