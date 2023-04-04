// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <string>
#include <popart/graph.hpp>
#include <popart/op/pow.hpp>
#include <popart/patterns/powarg1gradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/ir.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

bool PowArg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<PowArg1GradOp>();
}

// if (arg0 == 0 && arg1 >= 0 && arg1 != NaN)
//    tmp = 0
// else
//    tmp = out * log(arg_0)
// grad_out = grad_in * tmp
TensorId
PowArg1GradOpPattern::makeAllReplacementOps(Op *op,
                                            Ir *ir,
                                            const Tensor &gradIn,
                                            const Tensor &fwdIn0,
                                            const Tensor &fwdIn1,
                                            const Tensor &fwdOut) const {
  // create the new ops
  auto log            = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Log, op);
  auto mul1           = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto mul2           = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto in0eq0         = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Equal, op);
  auto in1lt0         = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Less, op);
  auto in1IsNan       = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::IsNaN, op);
  auto in1lt0orNan    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Or, op);
  auto in1ge0         = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Not, op);
  auto mul2InSelector = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::And, op);
  auto mul2InPicker   = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Where, op);

  // Create a 1-dim constant tensors with the same type as fwdIn0
  TensorInfo resultInfo(fwdIn0.info.dataType(), {1});
  auto zerosId = ir->createIntermediateTensorId(gradIn.id);
  addConstInitFromFloat(0.0f, zerosId, resultInfo, op->getGraph().getTensors());

  // Connect up the new ops
  log->connectInTensor(0, fwdIn0.id);
  log->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  log->outInfo(0) = log->inInfo(0);

  mul1->connectInTensor(0, fwdOut.id);
  mul1->connectInTensor(1, log->outTensor(0)->id);
  mul1->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  mul1->outInfo(0) = op->prettyNpOut(mul1->inInfo(0), mul1->inInfo(1));

  in0eq0->connectInTensor(0, fwdIn0.id);
  in0eq0->connectInTensor(1, zerosId);
  in0eq0->createAndConnectOutTensor(0,
                                    ir->createIntermediateTensorId(gradIn.id));
  in0eq0->setup();

  in1lt0->connectInTensor(0, fwdIn1.id);
  in1lt0->connectInTensor(1, zerosId);
  in1lt0->createAndConnectOutTensor(0,
                                    ir->createIntermediateTensorId(gradIn.id));
  in1lt0->setup();

  in1IsNan->connectInTensor(0, fwdIn1.id);
  in1IsNan->createAndConnectOutTensor(
      0, ir->createIntermediateTensorId(gradIn.id));
  in1IsNan->setup();

  in1lt0orNan->connectInTensor(0, in1lt0->outTensor(0)->id);
  in1lt0orNan->connectInTensor(1, in1IsNan->outTensor(0)->id);
  in1lt0orNan->createAndConnectOutTensor(
      0, ir->createIntermediateTensorId(gradIn.id));
  in1lt0orNan->setup();

  in1ge0->connectInTensor(0, in1lt0orNan->outTensor(0)->id);
  in1ge0->createAndConnectOutTensor(0,
                                    ir->createIntermediateTensorId(gradIn.id));
  in1ge0->setup();

  mul2InSelector->connectInTensor(0, in0eq0->outTensor(0)->id);
  mul2InSelector->connectInTensor(1, in1ge0->outTensor(0)->id);
  mul2InSelector->createAndConnectOutTensor(
      0, ir->createIntermediateTensorId(gradIn.id));
  mul2InSelector->setup();

  mul2InPicker->connectInTensor(0, mul2InSelector->outTensor(0)->id);
  mul2InPicker->connectInTensor(1, zerosId);
  mul2InPicker->connectInTensor(2, mul1->outTensor(0)->id);
  mul2InPicker->createAndConnectOutTensor(
      0, ir->createIntermediateTensorId(gradIn.id));
  mul2InPicker->setup();

  mul2->connectInTensor(0, gradIn.id);
  mul2->connectInTensor(1, mul2InPicker->outTensor(0)->id);
  mul2->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  mul2->outInfo(0) = op->prettyNpOut(mul2->inInfo(0), mul2->inInfo(1));

  return mul2->outTensor(0)->id;
}

namespace {
static PatternCreator<popart::PowArg1GradOpPattern>
    PowArg1GradOpPattern("PowArg1GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
