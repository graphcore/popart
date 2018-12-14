#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/patterns/cosgradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool CosGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<CosGradOp>();
}

std::vector<const Tensor *> CosGradOpPattern::touches(Op *) const { return {}; }

// grad_out = - grad_in * sin(fwd_in)
bool CosGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(CosGradOp::getGradInIndex());
  auto fwd_in   = op->inTensor(CosGradOp::getFwdArgInIndex());
  auto grad_out = op->outTensor(CosGradOp::getOutIndex());

  auto ir = op->pir;

  // create the new ops
  auto sin_op = make_unique<SinOp>(OpConstructorBundle{OpType::SIN, ir, {}});
  auto mul_op = make_unique<MulOp>(OpConstructorBundle{OpType::MUL, ir, {}});
  auto negate_op =
      make_unique<NegateOp>(OpConstructorBundle{OpType::NEGATE, ir, {}});

  // move ops into ir
  auto sin    = sin_op.get();
  auto mul    = mul_op.get();
  auto negate = negate_op.get();
  ir->moveIntoIr(std::move(sin_op));
  ir->moveIntoIr(std::move(mul_op));
  ir->moveIntoIr(std::move(negate_op));

  // Remove the CosGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

  // Connect up the new ops
  sin->connectInTensor(SinOp::getInIndex(), fwd_in->id);
  sin->createAndConnectOutTensor(SinOp::getOutIndex(),
                                 createImtermediateTensorId(grad_in->id));
  sin->setup();

  mul->connectInTensor(MulOp::getArg0InIndex(), grad_in->id);
  mul->connectInTensor(MulOp::getArg1InIndex(),
                       sin->outTensor(SinOp::getOutIndex())->id);
  mul->createAndConnectOutTensor(MulOp::getOutIndex(),
                                 createImtermediateTensorId(grad_in->id));
  mul->setup();

  negate->connectInTensor(NegateOp::getInIndex(),
                          mul->outTensor(MulOp::getOutIndex())->id);
  negate->connectOutTensor(NegateOp::getOutIndex(), grad_out->id);

  return true;
}

namespace {
static PatternCreator<CosGradOpPattern> CosGradOpPattern(PatternType::COSGRADOP,
                                                         "CosGradOp");
}

} // namespace poponnx
