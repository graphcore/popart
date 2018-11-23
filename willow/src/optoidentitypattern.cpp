#include <poponnx/identity.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/pad.hpp>
#include <poponnx/reducesum.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

#include <poponnx/optoidentitypattern.hpp>

namespace willow {

bool OpToIdentityPattern::matches(Op *op) const {
  // A reduce op that doesn't reduce anything
  return (op->isConvertibleTo<ReduceSumOp>() &&
          (op->input.tensor(0)->info.shape() ==
           op->output.tensor(0)->info.shape())) ||
         // A sum op with only one input
         (op->opType == OpType::SUM && op->input.n() == 1) ||
         // A pad op with no padding
         (op->opType == OpType::PAD &&
          dynamic_cast<const PadOp *>(op)->padSizeZero());
}

std::vector<const Tensor *> OpToIdentityPattern::touches(Op *) const {
  return {};
}

void OpToIdentityPattern::apply(Op *op) const {
  auto input_tensor  = op->input.tensor(0);
  auto output_tensor = op->output.tensor(0);
  auto ir            = op->pir;
  auto identity_op   = make_unique<IdentityOp>(
      OpConstructorBundle{"Identity", ir, {}, getOnnxDomain()});

  // Add the identity op to the IR
  auto identity = identity_op.get();
  ir->moveIntoIr(std::move(identity_op));

  // Remap the tensor-to-op relationships
  input_tensor->consumers.increment(identity);
  input_tensor->consumers.decrement(op);
  output_tensor->resetProducer(identity);

  // Remap the op-to-tensor relationships
  identity->input.insert(0, input_tensor);
  identity->output.insert(0, output_tensor);

  // Remove the op
  ir->eraseOp(op->id);
}

} // namespace willow
