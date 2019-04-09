#include <poponnx/device.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/relu.hpp>
#include <poponnx/popx/op/relux.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensorindex.hpp>

#include <popnn/NonLinearity.hpp>

namespace poponnx {
namespace popx {

ReluInplaceOpx::ReluInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, ReluComputex::get()) {
  verifyOp<ReluInplaceOp>(op, Onnx::CustomOperators::ReluInplace);
}

ReluOpx::ReluOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, ReluComputex::get()) {
  verifyOp<ReluOp>(op, Onnx::Operators::Relu_6);
}

poplar::Tensor ReluComputex::outplace(poplar::program::Sequence &p,
                                      poplar::Graph &g,
                                      const poplar::Tensor &t) const {
  auto outTensor = cloneNcopy(p, g, t);
  inplace(p, g, outTensor);
  return outTensor;
}

void ReluComputex::inplace(poplar::program::Sequence &p,
                           poplar::Graph &g,
                           const poplar::Tensor &t) const {

  // apply the inplace RELU
  popnn::nonLinearityInPlace(g, popnn::NonLinearityType::RELU, t, p, "");
}

ReluGradOpx::ReluGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReluGradOp>(op, Onnx::GradOperators::ReluGrad);
}

void ReluGradOpx::grow(poplar::program::Sequence &prog) const {

  ReluGradOp &rgop = getOp<ReluGradOp>();

  auto outTensor = popnn::nonLinearityInputGradient(
      graph(),                                 // graph,
      popnn::NonLinearityType::RELU,           // nonLinearityType,
      getInTensor(rgop.getReludInIndex()),     // out,
      getInTensor(rgop.getGradReludInIndex()), // outGradient,
      prog,                                    // prog,
      idStr()                                  // debugPrefix
  );

  setOutTensor(0, outTensor);
}

namespace {
OpxCreator<ReluOpx> reluxOpxCreator(Onnx::Operators::Relu_6);
OpxCreator<ReluInplaceOpx>
    reluxInplaceOpxCreator(Onnx::CustomOperators::ReluInplace);
OpxCreator<ReluGradOpx> reluxGradOpxCreator(Onnx::GradOperators::ReluGrad);
} // namespace

} // namespace popx
} // namespace poponnx
