// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <string>
#include <popart/op/div.hpp>
#include <popart/patterns/divarg0gradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/ir.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

bool DivArg0GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<DivArg0GradOp>();
}

// grad_out = grad_in / fwd_in1
TensorId
DivArg0GradOpPattern::makeAllReplacementOps(Op *op,
                                            Ir *ir,
                                            const Tensor &gradIn,
                                            const Tensor &fwdIn0,
                                            const Tensor &fwdIn1,
                                            const Tensor &fwdOut) const {
  // create the new ops
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);

  // Connect up the new ops
  div->connectInTensor(0, gradIn.id);
  div->connectInTensor(1, fwdIn1.id);
  div->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  div->outInfo(0) = op->prettyNpOut(gradIn.info, fwdIn1.info);

  return div->outTensor(0)->id;
}

namespace {
static PatternCreator<DivArg0GradOpPattern>
    DivArg0GradOpPattern("DivArg0GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
