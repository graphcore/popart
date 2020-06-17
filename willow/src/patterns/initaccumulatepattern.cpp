// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/add.hpp>
#include <popart/op/init.hpp>
#include <popart/op/pad.hpp>
#include <popart/patterns/initaccumulatepattern.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>

namespace popart {

bool InitAccumulatePattern::matches(Op *op) const {

  // Looking for element-wise binary op (two inputs checked).
  if (!op->isConvertibleTo<ElementWiseBinaryOp>() or op->input->n() != 2) {
    return false;
  }
  // Ignore ops for which inferTensorMappingToFrom has been modified already.
  if (!op->settings.inferTensorMappingToFrom.empty()) {
    return false;
  }

  const auto output = op->outTensor(AddOp::getOutIndex());

  const auto arg0Idx = ElementWiseBinaryOp::getArg0InIndex();
  const auto arg1Idx = ElementWiseBinaryOp::getArg1InIndex();

  const auto arg0 = op->inTensor(arg0Idx);
  const auto arg1 = op->inTensor(arg1Idx);
  if (!arg0->hasProducer() || !arg1->hasProducer()) {
    return false;
  }

  // Is the arg0 producer an Init op?
  const auto arg0Producer = arg0->getProducer();
  const auto arg0FromInit = arg0Producer->isConvertibleTo<InitOp>();

  // Is the arg1 producer an Init op?
  const auto arg1Producer = arg1->getProducer();
  const auto arg1FromInit = arg1Producer->isConvertibleTo<InitOp>();

  // One and only one argument should be from an InitOp.
  if (!(arg0FromInit ^ arg1FromInit)) {
    return false;
  }

  // Check for inplace modification (arg0).
  uint32_t mods = 0;
  for (auto reg : op->modifies(arg0Idx)) {
    if (!reg.isEmpty()) {
      ++mods;
    }
  }
  const auto inplace0 = (mods > 0);

  // Check for inplace modification (arg1).
  mods = 0;
  for (auto reg : op->modifies(arg1Idx)) {
    if (!reg.isEmpty()) {
      ++mods;
    }
  }
  const auto inplace1 = (mods > 0);

  // Ignore op if either operand is modified inplace.
  if (inplace0 | inplace1) {
    return false;
  }

  uint32_t ewbConsumers   = 0;
  uint32_t otherConsumers = 0;
  for (auto consumer : output->consumers.getOps()) {
    if (consumer->isConvertibleTo<ElementWiseBinaryOp>()) {
      ++ewbConsumers;
    } else {
      ++otherConsumers;
    }
  }

  // The output of this operation should be consumed by at least
  // one ElementWiseBinary op and no other op type.
  if (!ewbConsumers || otherConsumers) {
    return false;
  }

  return true;
}

std::vector<const Tensor *> InitAccumulatePattern::touches(Op *op) const {
  // This pattern affects the layout or relationship of both inputs.
  return {op->input->tensor(0), op->input->tensor(1)};
}

bool InitAccumulatePattern::apply(Op *op) const {
  const auto arg0Idx = ElementWiseBinaryOp::getArg0InIndex();
  const auto arg1Idx = ElementWiseBinaryOp::getArg1InIndex();

  const auto arg0         = op->inTensor(arg0Idx);
  const auto arg0Producer = arg0->getProducer();
  const auto arg0FromInit = arg0Producer->isConvertibleTo<InitOp>();

  if (arg0FromInit) {
    // Infer Arg0 layout from Arg1.
    op->settings.inferTensorMappingToFrom.insert({arg0Idx, arg1Idx});
  } else {
    // Infer Arg1 layout from Arg0.
    op->settings.inferTensorMappingToFrom.insert({arg1Idx, arg0Idx});
  }

  return true;
}

namespace {
static PatternCreator<InitAccumulatePattern>
    InitAccumulatePattern(PreAliasPatternType::InitAccumulate,
                          "InitAccumulate");
}

} // namespace popart
