// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/op/conv.hpp>
#include <popart/patterns/convflipweightsdoubleflippattern.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool ConvFlipWeightsDoubleFlipPattern::matches(Op *op) const {
  if (!op->isConvertibleTo<ConvFlipWeightsOp>()) {
    return false;
  }

  // Only match if this is the input is an ConvFlipWeightOp too
  auto input    = op->inTensor(ConvFlipWeightsOp::getInIndex());
  auto producer = input->getProducerUnsafe();
  if (!producer) {
    return false;
  }

  if (!producer->isConvertibleTo<ConvFlipWeightsOp>()) {
    return false;
  }

  return true;
}

std::vector<const Tensor *>
ConvFlipWeightsDoubleFlipPattern::touches(Op *op) const {
  // This tensor would be removed in favour of the become the unflipped
  // (rather than reflipped) output
  return {op->outTensor(ConvFlipWeightsOp::getOutIndex())};
}

bool ConvFlipWeightsDoubleFlipPattern::apply(Op *op) const {
  auto second_flip_in  = op->inTensor(ConvFlipWeightsOp::getInIndex());
  auto first_flip      = second_flip_in->getProducer();
  auto first_flip_in   = first_flip->inTensor(ConvFlipWeightsOp::getInIndex());
  auto second_flip_out = op->outTensor(ConvFlipWeightsOp::getOutIndex());

  op->disconnectAllInputs();
  op->disconnectAllOutputs();

  op->getGraph().replaceTensor(second_flip_out->id, first_flip_in->id);

  op->getGraph().eraseOp(op->id);
  op->getGraph().getTensors().remove(second_flip_out->id);

  return true;
}

namespace {
static PatternCreator<ConvFlipWeightsDoubleFlipPattern>
    ConvFlipWeightsDoubleFlipPattern("ConvFlipWeightsDoubleFlip", true, false);
}

} // namespace popart
