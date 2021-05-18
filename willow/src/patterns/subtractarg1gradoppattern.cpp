// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/subtract.hpp>
#include <popart/patterns/subtractarg1gradoppattern.hpp>
#include <popart/tensor.hpp>

namespace popart {

bool SubtractArg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<SubtractArg1GradOp>();
}

TensorId
SubtractArg1GradOpPattern::makeAllReplacementOps(Op *op,
                                                 Ir *ir,
                                                 const Tensor &gradIn,
                                                 const Tensor &fwdIn0,
                                                 const Tensor &fwdIn1,
                                                 const Tensor &fwdOut) const {
  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to SubtractArg1GradOp::matches
  auto &graph = op->getGraph();

  auto tmpOut = ir->createIntermediateTensorId(gradIn.id);

  graph.createConnectedOp<NegateOp>({{NegateOp::getInIndex(), gradIn.id}},
                                    {{NegateOp::getOutIndex(), tmpOut}},
                                    Onnx::AiOnnx::OpSet9::Neg,
                                    op->getSettings());

  return tmpOut;
}

namespace {
static PatternCreator<SubtractArg1GradOpPattern>
    PreUniReplPattern(PreAliasPatternType::SubtractArg1GradOp,
                      "SubtractArg1GradOp");
}

} // namespace popart
