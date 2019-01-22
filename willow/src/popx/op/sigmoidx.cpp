#include <popnn/NonLinearity.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sigmoid.hpp>
#include <poponnx/popx/op/sigmoidx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

SigmoidOpx::SigmoidOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SigmoidOp>(op, Onnx::Operators::Sigmoid_6);
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

SigmoidGradOpx::SigmoidGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SigmoidGradOp>(op, Onnx::GradOperators::SigmoidGrad);
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

namespace {
OpxCreator<SigmoidOpx> sigmoidOpxCreator(Onnx::Operators::Sigmoid_6);
OpxCreator<SigmoidGradOpx>
    sigmoidGradOpxCreator(Onnx::GradOperators::SigmoidGrad);
} // namespace

} // namespace popx
} // namespace poponnx
