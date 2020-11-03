// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/subtract.hpp>
#include <popart/patterns/subtractarg1gradoppattern.hpp>
#include <popart/tensor.hpp>

namespace popart {

bool SubtractArg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<SubtractArg1GradOp>();
}

std::vector<std::unique_ptr<Op>>
SubtractArg1GradOpPattern::sequence(Op *op) const {
  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to SubtractArg1GradOp::matches
  auto axes = dynamic_cast<SubtractArg1GradOp *>(op)->getReductionAxes();

  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Neg, op));
  seq.push_back(std::make_unique<ReduceSumOp>(
      Onnx::AiOnnx::OpSet9::ReduceSum, axes, false, op->getSettings()));

  return seq;
}

namespace {
static PatternCreator<SubtractArg1GradOpPattern>
    PreUniReplPattern(PreAliasPatternType::SubtractArg1GradOp,
                      "SubtractArg1GradOp");
}

} // namespace popart
