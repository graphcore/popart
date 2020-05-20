// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/op/add.hpp>
#include <popart/op/sum.hpp>
#include <popart/patterns/sumtoaddpattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool SumToAddPattern::matches(Op *op) const {
  return op->isConvertibleTo<SumOp>() && op->input->n() == 2;
}

std::vector<const Tensor *> SumToAddPattern::touches(Op *) const { return {}; }

// grad_out = grad_in / fwd_in1
bool SumToAddPattern::apply(Op *op) const {
  // have already verified that op has 2 inputs in ::matches, but
  // op->input->tensors() may return a vector of size 1. This happens if 1
  // tensor is used as multiple inputs to the sum op.
  auto lhs = op->inTensor(0);
  auto rhs = op->inTensor(1);
  auto out = op->outTensor(SumOp::getOutIndex());

  // create the new op and erase the old op
  auto add_op = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // connect the new op
  add_op->connectInTensor(AddOp::getArg0InIndex(), lhs->id);
  add_op->connectInTensor(AddOp::getArg1InIndex(), rhs->id);
  add_op->connectOutTensor(AddOp::getOutIndex(), out->id);

  return true;
}

namespace {
static PatternCreator<SumToAddPattern>
    SumToAddPattern(PreAliasPatternType::SumtoAdd, "SumToAdd");
}

} // namespace popart
