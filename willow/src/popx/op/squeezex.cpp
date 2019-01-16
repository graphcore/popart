#include <poponnx/error.hpp>
#include <poponnx/op/squeeze.hpp>
#include <poponnx/popx/op/squeezex.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
namespace popx {

void SqueezeOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, inId(SqueezeOp::getInIndex()));
  outTensor = outTensor.reshape(outInfo(SqueezeOp::getOutIndex()).shape_szt());
  insert(outId(SqueezeOp::getOutIndex()), outTensor);
}

SqueezeOpx::SqueezeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SqueezeOp>(op, {Onnx::Operators::Squeeze_1});
}

SqueezeGradOpx::SqueezeGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SqueezeGradOp>(op, Onnx::GradOperators::SqueezeGrad);
}

void SqueezeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, inId(SqueezeGradOp::getInIndex()));
  outTensor = outTensor.reshape(outInfo(SqueezeOp::getOutIndex()).shape_szt());
  insert(outId(SqueezeGradOp::getOutIndex()), outTensor);
}

namespace {
OpxCreator<SqueezeOpx> squeezeOpxCreator(Onnx::Operators::Squeeze_1);
OpxCreator<SqueezeGradOpx>
    squeezeGradOpxCreator(Onnx::GradOperators::SqueezeGrad);
} // namespace

} // namespace popx
} // namespace poponnx
