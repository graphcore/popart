#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/subsample.hpp>
#include <poponnx/patterns/optoidentitypattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

bool OpToIdentityPattern::matches(Op *op) const {
  // A reduce op that doesn't reduce anything
  return ((op->isConvertibleTo<ReduceSumOp>() &&
           (op->input->tensor(0)->info.shape() ==
            op->output->tensor(0)->info.shape())) ||
          // A sum op with only one input
          //(op->opType == OpType::SUM && op->input->n() == 1) ||
          (op->opid == Onnx::Operators::Sum_6 && op->input->n() == 1) ||
          (op->opid == Onnx::Operators::Sum_8 && op->input->n() == 1) ||
          // A pad op with no padding
          //(op->opType == OpType::PAD && dynamic_cast<const PadOp
          //*>(op)->padSizeZero()) ||
          (op->opid == Onnx::Operators::Pad_2 &&
           dynamic_cast<const PadOp *>(op)->padSizeZero()) ||
          // A subsample with all strides being 1
          (op->opid == Onnx::CustomOperators::Subsample_1 &&
           dynamic_cast<const SubsampleOp *>(op)->strideSizeOne()) ||
          // Concat a single tensor
          (op->opid == Onnx::Operators::Concat_4 && op->input->n() == 1) ||
          // Inplace concat a single tensor
          (op->opid == Onnx::CustomOperators::ConcatInplace &&
           op->input->n() == 1));
}

std::vector<const Tensor *> OpToIdentityPattern::touches(Op *) const {
  return {};
}

bool OpToIdentityPattern::apply(Op *op) const {
  auto input_tensor  = op->input->tensor(0);
  auto output_tensor = op->output->tensor(0);
  auto ir            = op->pir;
  auto attr          = op->nAtts.filter(sVirtualGraphAttribute);

  auto identity_op = make_unique<IdentityOp>(
      Onnx::AiOnnx::OpSet9::Identity, ir, std::string{}, attr);

  // Add the identity op to the IR
  auto identity = identity_op.get();
  ir->moveIntoIr(std::move(identity_op));

  // Remap the tensor-to-op relationships
  input_tensor->consumers.increment(identity);
  input_tensor->consumers.decrement(op);
  output_tensor->resetProducer(identity);

  // Remap the op-to-tensor relationships
  identity->input->insert(0, input_tensor);
  identity->output->insert(0, output_tensor);

  // Remove the op
  ir->eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<OpToIdentityPattern>
    opToIdentityPattern(PatternType::OPTOIDENTITY, "OpToIdentity");
}

} // namespace poponnx
