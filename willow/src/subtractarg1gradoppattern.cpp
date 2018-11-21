#include <poponnx/identity.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/reducesum.hpp>
#include <poponnx/subtract.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

#include <poponnx/subtractarg1gradoppattern.hpp>

namespace willow {

bool SubtractArg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<SubtractArg1GradOp>();
}

std::vector<const Tensor *> SubtractArg1GradOpPattern::touches(Op *) const {
  return {};
}

void SubtractArg1GradOpPattern::apply(Op *op) const {
  auto input_tensor  = op->input.tensor(0);
  auto output_tensor = op->output.tensor(0);
  auto ir            = op->pir;
  auto negate_op     = make_unique<NegateOp>(
      OpConstructorBundle{"Negate", ir, {}, getPoponnxDomain()});
  auto axes =
      npReductionAxis(output_tensor->info.shape(), input_tensor->info.shape());
  auto reducesum_op = make_unique<ReduceSumOp>(
      OpConstructorBundle{"ReduceSum", ir, {}, getPoponnxDomain()},
      axes,
      false);

  const auto tmp_tensor_id = "t__" + op->output.id(0);
  op->pir->getTensors().addActGrad(tmp_tensor_id);
  const auto tmp_tensor = ir->getTensors().get(tmp_tensor_id);
  tmp_tensor->info      = input_tensor->info;

  auto negate = negate_op.get();
  ir->moveIntoIr(std::move(negate_op));

  auto reducesum = reducesum_op.get();
  ir->moveIntoIr(std::move(reducesum_op));

  // Remap the tensor-to-op relationships
  input_tensor->consumers.increment(negate);
  input_tensor->consumers.decrement(op);
  tmp_tensor->consumers.increment(reducesum);
  tmp_tensor->setProducer(negate);
  output_tensor->resetProducer(reducesum);

  // Remap the op-to-tensor relationships
  negate->input.insert(0, input_tensor);
  negate->output.insert(0, tmp_tensor);
  reducesum->input.insert(0, tmp_tensor);
  reducesum->output.insert(0, output_tensor);

  // Remove the reducesum op
  ir->eraseOp(op->id);
}

} // namespace willow
