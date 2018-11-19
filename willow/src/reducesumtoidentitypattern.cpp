#include <poponnx/identity.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/reducesum.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

#include <poponnx/reducesumtoidentitypattern.hpp>

namespace willow {

bool ReduceSumToIdentityPattern::matches(Op *op) const {
  // A reduce sum that doesn't change the shape is an identity operation
  return op->isConvertibleTo<ReduceSumOp>() &&
         (op->input.tensor(0)->info.shape() ==
          op->output.tensor(0)->info.shape());
}

std::vector<const Tensor *> ReduceSumToIdentityPattern::touches(Op *) const {
  return {};
}

void ReduceSumToIdentityPattern::apply(Op *op) const {
  auto reduce        = dynamic_cast<ReduceSumOp *>(op);
  auto input_tensor  = reduce->input.tensor(0);
  auto output_tensor = reduce->output.tensor(0);
  auto ir            = op->pir;
  auto identity_op   = make_unique<IdentityOp>(
      OpConstructorBundle{"Identity", ir, {}, getPoponnxDomain()});

  // Add the identity op to the IR
  auto identity = identity_op.get();
  ir->moveIntoIr(std::move(identity_op));

  // Remap the tensor-to-op relationships
  input_tensor->consumers.increment(identity);
  input_tensor->consumers.decrement(reduce);
  output_tensor->resetProducer(identity);

  // Remap the op-to-tensor relationships
  identity->input.insert(0, input_tensor);
  identity->output.insert(0, output_tensor);

  // Remove the reducesum op
  ir->eraseOp(reduce->id);
}

} // namespace willow
