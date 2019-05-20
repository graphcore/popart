#include <popnn/NonLinearity.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sigmoid.hpp>
#include <poponnx/popx/op/sigmoidx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

SigmoidInplaceOpx::SigmoidInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, SigmoidComputex::get()) {
  verifyOp<SigmoidInplaceOp>(op, Onnx::CustomOperators::SigmoidInplace);
}

SigmoidOpx::SigmoidOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, SigmoidComputex::get()) {
  verifyOp<SigmoidOp>(op, Onnx::Operators::Sigmoid_6);
}

poplar::Tensor SigmoidComputex::outplace(poplar::program::Sequence &p,
                                         poplar::Graph &g,
                                         const poplar::Tensor &t,
                                         const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t);
  inplace(p, g, outTensor, s);
  return outTensor;
}

void SigmoidComputex::inplace(poplar::program::Sequence &p,
                              poplar::Graph &g,
                              const poplar::Tensor &t,
                              const std::string &s) const {

  // apply the inplace SIGMOID
  popnn::nonLinearityInPlace(g, popnn::NonLinearityType::SIGMOID, t, p, s);
}

SigmoidGradOpx::SigmoidGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SigmoidGradOp>(op, Onnx::GradOperators::SigmoidGrad);
}

void SigmoidGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = popnn::nonLinearityInputGradient(
      graph(),                                        // graph,
      popnn::NonLinearityType::SIGMOID,               // nonLinearityType,
      getInTensor(SigmoidGradOp::getFwdOutInIndex()), // out,
      getInTensor(SigmoidGradOp::getGradInIndex()),   // outGradient,
      prog,                                           // prog,
      idStr()                                         // debugPrefix
  );

  setOutTensor(SigmoidOp::getOutIndex(), outTensor);
}

namespace {
OpxCreator<SigmoidOpx> sigmoidOpxCreator(Onnx::Operators::Sigmoid_6);
OpxCreator<SigmoidGradOpx>
    sigmoidGradOpxCreator(Onnx::GradOperators::SigmoidGrad);
OpxCreator<SigmoidInplaceOpx>
    sigmoidxInplaceOpxCreator(Onnx::CustomOperators::SigmoidInplace);
} // namespace

} // namespace popx
} // namespace poponnx
