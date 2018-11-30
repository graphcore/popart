#include <utility>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/op/reciprocal.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/patterns/reciprocalgradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool ReciprocalGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<ReciprocalGradOp>();
}

std::vector<const Tensor *> ReciprocalGradOpPattern::touches(Op *) const {
  return {};
}

bool ReciprocalGradOpPattern::apply(Op *op) const {
  auto input  = op->input->tensor(0);
  auto output = op->output->tensor(0);
  auto ir     = op->pir;

  // create the new ops
  auto square_op     = make_unique<SquareOp>(OpConstructorBundle{
      "Square", ir, {}, getOpTypes().getDomain(OpType::SQUARE)});
  auto reciprocal_op = make_unique<ReciprocalOp>(OpConstructorBundle{
      "Reciprocal", ir, {}, getOpTypes().getDomain(OpType::RECIPROCAL)});
  auto negate_op     = make_unique<NegateOp>(OpConstructorBundle{
      "Negate", ir, {}, getOpTypes().getDomain(OpType::NEGATE)});

  // move ops into ir
  auto square     = square_op.get();
  auto reciprocal = reciprocal_op.get();
  auto negate     = negate_op.get();
  ir->moveIntoIr(std::move(square_op));
  ir->moveIntoIr(std::move(reciprocal_op));
  ir->moveIntoIr(std::move(negate_op));

  // create a tensors to connect the new ops
  const auto square_reciprocal_tensor_id = "t__0__" + op->output->id(0);
  op->pir->getTensors().addActGrad(square_reciprocal_tensor_id);
  const auto square_reciprocal_tensor =
      ir->getTensors().get(square_reciprocal_tensor_id);
  square_reciprocal_tensor->info = input->info;

  const auto reciprocal_negate_tensor_id = "t__1__" + op->output->id(0);
  op->pir->getTensors().addActGrad(reciprocal_negate_tensor_id);
  const auto reciprocal_negate_tensor =
      ir->getTensors().get(reciprocal_negate_tensor_id);
  reciprocal_negate_tensor->info = input->info;

  // Remap the tensor-to-op relationships
  input->consumers.decrement(op);
  input->consumers.increment(square);

  square_reciprocal_tensor->setProducer(square);
  square_reciprocal_tensor->consumers.increment(reciprocal);
  reciprocal_negate_tensor->setProducer(reciprocal);
  reciprocal_negate_tensor->consumers.increment(negate);

  output->resetProducer(negate);

  // Remap the op-to-tensor relationships
  square->input->insert(0, input);
  square->output->insert(0, square_reciprocal_tensor);
  reciprocal->input->insert(0, square_reciprocal_tensor);
  reciprocal->output->insert(0, reciprocal_negate_tensor);
  negate->input->insert(0, reciprocal_negate_tensor);
  negate->output->insert(0, output);

  // Remove the reducesum op
  ir->eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<ReciprocalGradOpPattern>
    reciprocalGradOpPattern(PatternType::RECIPROCALGRADOP, "ReciprocalGradOp");
}

} // namespace poponnx
