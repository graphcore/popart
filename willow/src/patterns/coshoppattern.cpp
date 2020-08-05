// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/cosh.hpp>
#include <popart/op/exp.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/scale.hpp>
#include <popart/patterns/coshoppattern.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool CoshOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<CoshOp>();
}

std::vector<const Tensor *> CoshOpPattern::touches(Op *) const { return {}; }

// output = (exp(input) + exp(-input)) * 0.5
bool CoshOpPattern::apply(Op *op) const {
  auto input  = op->inTensor(CoshOp::getInIndex());
  auto output = op->outTensor(CoshOp::getOutIndex());

  // create the new ops
  auto negate = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Neg, op);
  auto exp1   = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Exp, op);
  auto exp2   = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Exp, op);
  auto add    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);

  auto scale = dynamic_cast<ScaleOp *>(
      makeReplacementOpInIr(Onnx::CustomOperators::Scale_1, op));
  scale->setScaleFactor(0.5f);

  // Remove the CoshOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  exp1->connectInTensor(ExpOp::getInIndex(), input->id);
  exp1->createAndConnectOutTensor(
      ExpOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  exp1->setup();

  negate->connectInTensor(NegateOp::getInIndex(), input->id);
  negate->createAndConnectOutTensor(
      NegateOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  negate->setup();

  exp2->connectInTensor(ExpOp::getInIndex(),
                        negate->outTensor(NegateOp::getOutIndex())->id);
  exp2->createAndConnectOutTensor(
      ExpOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  exp2->setup();

  add->connectInTensor(AddOp::getArg0InIndex(),
                       exp1->outTensor(ExpOp::getOutIndex())->id);
  add->connectInTensor(AddOp::getArg1InIndex(),
                       exp2->outTensor(ExpOp::getOutIndex())->id);
  add->createAndConnectOutTensor(
      AddOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  add->setup();

  scale->connectInTensor(ScaleOp::getInIndex(),
                         add->outTensor(ScaleOp::getOutIndex())->id);
  scale->connectOutTensor(ScaleOp::getOutIndex(), output->id);

  return true;
}

namespace {
static PatternCreator<CoshOpPattern> CoshOpPattern(PreAliasPatternType::CoshOp,
                                                   "CoshOp",
                                                   /* enabled = */ true,
                                                   /* mandatory = */ true);
}

} // namespace popart
