#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/subsample.hpp>
#include <poponnx/opmanager.hpp>
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

std::vector<std::unique_ptr<Op>> OpToIdentityPattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Identity, op, {}));

  return seq;
}

namespace {
static PatternCreator<OpToIdentityPattern>
    opToIdentityPattern(PatternType::OPTOIDENTITY, "OpToIdentity");
}

} // namespace poponnx
