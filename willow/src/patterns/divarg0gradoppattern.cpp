// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op/div.hpp>
#include <popart/patterns/divarg0gradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool DivArg0GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<DivArg0GradOp>();
}

// grad_out = grad_in / fwd_in1
TensorId DivArg0GradOpPattern::makeAllReplacementOps(Op *op,
                                                     Tensor *grad_in,
                                                     Tensor *fwd_in0,
                                                     Tensor *fwd_in1,
                                                     Tensor *fwd_out) const {
  // create the new ops
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);

  // Connect up the new ops
  div->connectInTensor(0, grad_in->id);
  div->connectInTensor(1, fwd_in1->id);
  div->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  div->outInfo(0) = op->prettyNpOut(grad_in->info, fwd_in1->info);

  return div->outTensor(0)->id;
}

namespace {
static PatternCreator<DivArg0GradOpPattern>
    DivArg0GradOpPattern(PreAliasPatternType::DivArg0GradOp,
                         "DivArg0GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
