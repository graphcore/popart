#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/cosh.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/patterns/tanhgradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool TanhGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<TanhGradOp>();
}

std::vector<const Tensor *> TanhGradOpPattern::touches(Op *) const {
  return {};
}

// grad_out = grad_in / square(cosh(fwd_in))
bool TanhGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(TanhGradOp::getGradInIndex());
  auto fwd_in   = op->inTensor(TanhGradOp::getFwdArgInIndex());
  auto grad_out = op->outTensor(TanhGradOp::getOutIndex());

  auto ir = op->pir;

  // create the new ops
  auto cosh_op = make_unique<CoshOp>(OpConstructorBundle{OpType::COSH, ir, {}});
  auto square_op =
      make_unique<SquareOp>(OpConstructorBundle{OpType::SQUARE, ir, {}});
  auto div_op = make_unique<DivOp>(OpConstructorBundle{OpType::DIV, ir, {}});

  // move ops into ir
  auto cosh   = cosh_op.get();
  auto square = square_op.get();
  auto div    = div_op.get();
  ir->moveIntoIr(std::move(cosh_op));
  ir->moveIntoIr(std::move(square_op));
  ir->moveIntoIr(std::move(div_op));

  // Remove the TanhGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

  // Connect up the new ops
  cosh->connectInTensor(CoshOp::getInIndex(), fwd_in->id);
  cosh->createAndConnectOutTensor(CoshOp::getOutIndex(),
                                  createTemporaryTensorId(grad_in->id));
  cosh->setup();

  square->connectInTensor(SquareOp::getInIndex(),
                          cosh->outTensor(CoshOp::getOutIndex())->id);
  square->createAndConnectOutTensor(SquareOp::getOutIndex(),
                                    createTemporaryTensorId(grad_in->id));
  square->setup();

  div->connectInTensor(DivOp::getArg0InIndex(), grad_in->id);
  div->connectInTensor(DivOp::getArg1InIndex(),
                       square->outTensor(SquareOp::getOutIndex())->id);
  div->connectOutTensor(DivOp::getOutIndex(), grad_out->id);

  return true;
}

namespace {
static PatternCreator<TanhGradOpPattern>
    TanhGradOpPattern(PatternType::TANHGRADOP, "TanhGradOp");
}

} // namespace poponnx
