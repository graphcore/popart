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

std::vector<std::unique_ptr<Op>>
SubtractArg1GradOpPattern::sequence(Op *op) const {
  auto ir            = op->pir;
  auto attr          = op->nAtts.filter(sVirtualGraphAttribute);
  auto input_tensor  = op->input->tensor(0);
  auto output_tensor = op->output->tensor(0);
  auto axes =
      npReductionAxis(output_tensor->info.shape(), input_tensor->info.shape());

  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Neg, op, {}));
  seq.push_back(make_unique<ReduceSumOp>(
      Onnx::AiOnnx::OpSet9::ReduceSum, ir, axes, false, attr));

  return seq;
}

namespace {
static PatternCreator<SubtractArg1GradOpPattern>
    PreUniReplPattern(PatternType::SUBTRACTARG1GRADOP, "SubtractArg1GradOp");
}

} // namespace poponnx
