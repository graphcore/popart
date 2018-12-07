#include <popnn/NonLinearity.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sigmoid.hpp>
#include <poponnx/popx/op/sigmoidx.hpp>

namespace poponnx {
namespace popx {

SigmoidOpx::SigmoidOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<SigmoidOp>()) {
    throw error("cannot create SigmoidOpx from " + op->op_type());
  }
}

void SigmoidOpx::grow(poplar::program::Sequence &prog) const {
  // There is only an in-place popnn Tanh. We therefore clone first,
  auto outTensor = cloneNcopy(prog, inId(SigmoidOp::getInIndex()));

  // and apply the inplace relu.
  popnn::nonLinearityInPlace(graph(),
                             popnn::NonLinearityType::SIGMOID,
                             outTensor,
                             prog,
                             outId(SigmoidOp::getOutIndex()));

  insert(outId(SigmoidOp::getOutIndex()), outTensor);
}

SigmoidGradOpx::SigmoidGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<SigmoidGradOp>()) {
    throw error("cannot create SigmoidGradOpx from " + op->op_type());
  }
}

void SigmoidGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = popnn::nonLinearityInputGradient(
      graph(),                                      // graph,
      popnn::NonLinearityType::SIGMOID,             // nonLinearityType,
      get(inId(SigmoidGradOp::getFwdOutInIndex())), //  out,
      get(inId(SigmoidGradOp::getGradInIndex())),   //  outGradient,
      prog,                                         // prog,
      idStr()                                       // debugPrefix
  );

  insert(outId(SigmoidOp::getOutIndex()), outTensor);
}

} // namespace popx
} // namespace poponnx
