#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/patterns/mularggradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool MulArgGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<MulArgGradOp>();
}

std::vector<const Tensor *> MulArgGradOpPattern::touches(Op *) const {
  return {};
}

bool MulArgGradOpPattern::apply(Op *op) const {
  auto input_0 = op->input->tensor(0);
  auto input_1 = op->input->tensor(1);
  auto output  = op->output->tensor(0);

  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to MulArgGradOpPattern::matches
  auto axes = dynamic_cast<MulArgGradOp *>(op)->getReductionAxes();

  auto ir = op->pir;

  // create the new ops
  auto mul_op = make_unique<MulOp>(OpConstructorBundle{OpType::MUL, ir, {}});
  auto reduce_sum_op = make_unique<ReduceSumOp>(
      OpConstructorBundle{OpType::REDUCESUM, ir, {}}, axes, false);

  // move ops into ir
  auto mul        = mul_op.get();
  auto reduce_sum = reduce_sum_op.get();
  ir->moveIntoIr(std::move(mul_op));
  ir->moveIntoIr(std::move(reduce_sum_op));

  // create a tensor to connect the multiply and reducesum ops
  const auto tmp_tensor_id = "t__" + op->output->id(0);
  op->pir->getTensors().addActGrad(tmp_tensor_id);
  const auto tmp_tensor = ir->getTensors().get(tmp_tensor_id);
  tmp_tensor->info      = npOut(input_0->info, input_1->info);

  // Remap the tensor-to-op relationships
  input_0->consumers.decrement(op);
  input_0->consumers.increment(mul);

  input_1->consumers.decrement(op);
  input_1->consumers.increment(mul);

  tmp_tensor->setProducer(mul);
  tmp_tensor->consumers.increment(reduce_sum);

  output->resetProducer(reduce_sum);

  // Remap the op-to-tensor relationships
  mul->input->insert(0, input_0);
  mul->input->insert(1, input_1);
  mul->output->insert(0, tmp_tensor);
  reduce_sum->input->insert(0, tmp_tensor);
  reduce_sum->output->insert(0, output);

  // Remove the reducesum op
  ir->eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<MulArgGradOpPattern>
    MulArgGradOpPattern(PatternType::MULARGGRADOP, "MulArgGradOp");
}

} // namespace poponnx
