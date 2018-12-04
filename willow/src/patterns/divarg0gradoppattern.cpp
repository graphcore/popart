#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/patterns/divarg0gradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool DivArg0GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<DivArg0GradOp>();
}

std::vector<const Tensor *> DivArg0GradOpPattern::touches(Op *) const {
  return {};
}

// grad_out = grad_in / fwd_in1
bool DivArg0GradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(DivArg0GradOp::getGradInIndex());
  auto fwd_in1  = op->inTensor(DivArg0GradOp::getFwdArg0InIndex());
  auto grad_out = op->outTensor(DivArg0GradOp::getOutIndex());

  auto ir = op->pir;

  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to DivArg0GradOpPattern::matches
  auto axes = dynamic_cast<DivArg0GradOp *>(op)->getReductionAxes();

  // create the new ops
  auto div_op = make_unique<DivOp>(
      OpConstructorBundle{"Div", ir, {}, getOpTypes().getDomain(OpType::DIV)});
  auto reduce_op = make_unique<ReduceSumOp>(
      OpConstructorBundle{
          "ReduceSum", ir, {}, getOpTypes().getDomain(OpType::REDUCESUM)},
      axes,
      false);

  // move ops into ir
  auto div    = div_op.get();
  auto reduce = reduce_op.get();
  ir->moveIntoIr(std::move(div_op));
  ir->moveIntoIr(std::move(reduce_op));

  // Remove the DivArg0GradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

  // Connect up the new ops
  div->connectInTensor(0, grad_in->id);
  div->connectInTensor(1, fwd_in1->id);
  div->createAndConnectOutTensor(0, "t__" + grad_in->id);
  div->outInfo(0) = npOut(grad_in->info, fwd_in1->info);

  reduce->connectInTensor(0, div->outTensor(0)->id);
  reduce->connectOutTensor(0, grad_out->id);

  return true;
}

namespace {
static PatternCreator<DivArg0GradOpPattern>
    DivArg0GradOpPattern(PatternType::DIVARG0GRADOP, "DivArg0GradOp");
}

} // namespace poponnx
