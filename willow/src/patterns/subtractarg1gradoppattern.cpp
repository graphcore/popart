#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/subtract.hpp>
#include <poponnx/patterns/subtractarg1gradoppattern.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

bool SubtractArg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<SubtractArg1GradOp>();
}

std::vector<const Tensor *> SubtractArg1GradOpPattern::touches(Op *) const {
  return {};
}

bool SubtractArg1GradOpPattern::apply(Op *op) const {
  auto input_tensor  = op->input->tensor(0);
  auto output_tensor = op->output->tensor(0);

  auto reducesum = dynamic_cast<ReduceSumOp *>(
      makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::ReduceSum, op));
  reducesum->setAxes(
      npReductionAxis(output_tensor->info.shape(), input_tensor->info.shape()));
  // do not keep reduced dims
  reducesum->setKeepDims(0l);

  auto negate = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Neg, op);

  const auto tmp_tensor_id = createIntermediateTensorId(op->output->id(0));
  op->pir->getTensors().addActGrad(tmp_tensor_id);
  const auto tmp_tensor = op->pir->getTensors().get(tmp_tensor_id);
  tmp_tensor->info      = input_tensor->info;

  // Remap the tensor-to-op relationships
  input_tensor->consumers.increment(negate);
  input_tensor->consumers.decrement(op);
  tmp_tensor->consumers.increment(reducesum);
  tmp_tensor->setProducer(negate);
  output_tensor->resetProducer(reducesum);

  // Remap the op-to-tensor relationships
  negate->input->insert(0, input_tensor);
  negate->output->insert(0, tmp_tensor);
  reducesum->input->insert(0, tmp_tensor);
  reducesum->output->insert(0, output_tensor);

  // Remove the reducesum op
  op->pir->eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<SubtractArg1GradOpPattern>
    PreUniReplPattern(PatternType::SUBTRACTARG1GRADOP, "SubtractArg1GradOp");
}

} // namespace poponnx
