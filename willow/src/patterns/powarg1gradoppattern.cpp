#include <memory>
#include <popart/graph.hpp>
#include <popart/op/log.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/pow.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/patterns/powarg1gradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool PowArg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<PowArg1GradOp>();
}

std::vector<const Tensor *> PowArg1GradOpPattern::touches(Op *) const {
  return {};
}

// grad_out = grad_in * out * log(arg_0)
bool PowArg1GradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(PowArg1GradOp::getGradInIndex());
  auto fwd_in0  = op->inTensor(PowArg1GradOp::getFwdArg0InIndex());
  auto out      = op->inTensor(PowArg1GradOp::getFwdOutIndex());
  auto grad_out = op->outTensor(PowArg1GradOp::getOutIndex());
  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to PowArg1GradOpPattern::matches
  auto axes = dynamic_cast<PowArg1GradOp *>(op)->getReductionAxes();

  // create the new ops
  auto log    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Log, op);
  auto mul_1  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto mul_2  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto reduce = dynamic_cast<ReduceSumOp *>(
      makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::ReduceSum, op));

  reduce->setAxes(axes);
  // do not keep reduced dims
  reduce->setKeepDims(0l);

  // Remove the PowArg1GradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  log->connectInTensor(0, fwd_in0->id);
  log->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  log->outInfo(0) = log->inInfo(0);

  mul_1->connectInTensor(0, out->id);
  mul_1->connectInTensor(1, log->outTensor(0)->id);
  mul_1->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  mul_1->outInfo(0) = npOut(mul_1->inInfo(0), mul_1->inInfo(1));

  mul_2->connectInTensor(0, grad_in->id);
  mul_2->connectInTensor(1, mul_1->outTensor(0)->id);
  mul_2->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  mul_2->outInfo(0) = npOut(mul_2->inInfo(0), mul_2->inInfo(1));

  reduce->connectInTensor(0, mul_2->outTensor(0)->id);
  reduce->connectOutTensor(0, grad_out->id);

  return true;
}

namespace {
static PatternCreator<PowArg1GradOpPattern>
    PowArg1GradOpPattern(PreAliasPatternType::POWARG1GRADOP, "PowArg1GradOp");
}

} // namespace popart
