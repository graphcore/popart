#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/gather.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/op/subsample.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/patterns/optoidentitypattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

bool OpToIdentityPattern::matches(Op *op) const {
  // TODO : T6634  Replace the if statements with a virtual call to the op to
  // determine if it can be replaced by the identity op

  // A reduce op that doesn't reduce anything

  if (op->isConvertibleTo<ReduceSumOp>() &&
      (op->input->tensor(0)->info.shape() ==
       op->output->tensor(0)->info.shape())) {
    return true;
  }

  // A sum op with only one input
  if (op->opid == Onnx::Operators::Sum_6 && op->input->n() == 1) {
    return true;
  }

  if (op->opid == Onnx::Operators::Sum_8 && op->input->n() == 1) {
    return true;
  }
  // A pad op with no padding
  auto pad = dynamic_cast<const PadOp *>(op);
  if (op->opid == Onnx::Operators::Pad_2 && pad->padSizeZero()) {
    return true;
  }

  // A subsample with all strides being 1
  auto subsample = dynamic_cast<const SubsampleOp *>(op);
  if (op->opid == Onnx::CustomOperators::Subsample_1 &&
      subsample->strideSizeOne()) {
    return true;
  }

  // Concat a single tensor
  if (op->opid == Onnx::Operators::Concat_4 && op->input->n() == 1) {
    return true;
  }

  // Inplace concat a single tensor
  if (op->opid == Onnx::CustomOperators::ConcatInplace && op->input->n() == 1) {
    return true;
  }

  // Dropout in testing mode can be replaced by the identity
  if (!op->getIr().isTraining()) {
    if (op->opid == Onnx::Operators::Dropout_6 ||
        op->opid == Onnx::Operators::Dropout_7) {
      return true;
    }
  }

  // A gather on a degenerate dimension with a rank 1 index tensor with a single
  // element
  auto gather = dynamic_cast<const GatherOp *>(op);
  if (op->opid == Onnx::Operators::Gather_1 &&
      gather->inShape(GatherOp::dataInIndex())[gather->getAxis()] == 1 &&
      gather->inInfo(GatherOp::indicesInIndex()).rank() == 1 &&
      gather->inInfo(GatherOp::indicesInIndex()).nelms() == 1) {
    return true;
  }

  // A scale with a scale factor of +1
  auto scale_op = dynamic_cast<const ScaleOp *>(op);
  if (op->opid == Onnx::Operators::Scale_1 &&
      scale_op->getScaleFactor() == 1.0f) {
    return true;
  }

  return false;
}

std::vector<std::unique_ptr<Op>> OpToIdentityPattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Identity, op, {}));

  return seq;
}

namespace {
static PatternCreator<OpToIdentityPattern>
    opToIdentityPattern(PreAliasPatternType::OPTOIDENTITY, "OpToIdentity");
}

} // namespace poponnx
