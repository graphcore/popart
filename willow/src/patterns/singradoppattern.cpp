#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/patterns/singradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool SinGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<SinGradOp>();
}

std::vector<const Tensor *> SinGradOpPattern::touches(Op *) const { return {}; }

// grad_out = grad_in * cos(fwd_in)
bool SinGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(SinGradOp::getGradInIndex());
  auto fwd_in   = op->inTensor(SinGradOp::getFwdArgInIndex());
  auto grad_out = op->outTensor(SinGradOp::getOutIndex());

  auto ir = op->pir;

  // create the new ops
  auto cos_op = make_unique<CosOp>(Onnx::Operators::Cos, ir);
  auto mul_op = make_unique<MulOp>(Onnx::Operators::Mul, ir);

  // move ops into ir
  auto cos = cos_op.get();
  auto mul = mul_op.get();
  ir->moveIntoIr(std::move(cos_op));
  ir->moveIntoIr(std::move(mul_op));

  // Remove the SinGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

  // Connect up the new ops
  cos->connectInTensor(CosOp::getInIndex(), fwd_in->id);
  cos->createAndConnectOutTensor(CosOp::getOutIndex(),
                                 createImtermediateTensorId(grad_in->id));
  cos->setup();

  mul->connectInTensor(MulOp::getArg0InIndex(), grad_in->id);
  mul->connectInTensor(MulOp::getArg1InIndex(),
                       cos->outTensor(CosOp::getOutIndex())->id);
  mul->connectOutTensor(MulOp::getOutIndex(), grad_out->id);

  return true;
}

namespace {
static PatternCreator<SinGradOpPattern> SinGradOpPattern(PatternType::SINGRADOP,
                                                         "SinGradOp");
}

} // namespace poponnx
