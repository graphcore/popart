#include <poponnx/error.hpp>
#include <poponnx/op/reshape.hpp>
#include <poponnx/popx/op/reshapex.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
namespace popx {

// Test note : scale by 1.0001 in grad op makes the test fail. Good.
void ReshapeOpx::grow(poplar::program::Sequence &prog) const {
  // not in-place, so cloning input
  auto outTensor = cloneNcopy(prog, inId(ReshapeOp::getInIndex()));
  outTensor = outTensor.reshape(outInfo(ReshapeOp::getOutIndex()).shape_szt());
  insert(outId(ReshapeOp::getOutIndex()), outTensor);
}

ReshapeOpx::ReshapeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReshapeOp>(op);
}

ReshapeGradOpx::ReshapeGradOpx(Op *op, Devicex *devicex)
    : ReshapeOpx(op, devicex) {
  verifyOp<ReshapeGradOp>(op, Onnx::GradOperators::ReshapeGrad);
}

namespace {
OpxCreator<ReshapeOpx> reshapeOpxCreator(Onnx::Operators::Reshape_5);
OpxCreator<ReshapeGradOpx>
    reshapeGradOpxCreator(Onnx::GradOperators::ReshapeGrad);
} // namespace

} // namespace popx
} // namespace poponnx
