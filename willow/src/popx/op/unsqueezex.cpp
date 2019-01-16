#include <poponnx/error.hpp>
#include <poponnx/op/unsqueeze.hpp>
#include <poponnx/popx/op/unsqueezex.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
namespace popx {

void UnsqueezeOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, inId(UnsqueezeOp::getInIndex()));
  outTensor =
      outTensor.reshape(outInfo(UnsqueezeOp::getOutIndex()).shape_szt());
  insert(outId(UnsqueezeOp::getOutIndex()), outTensor);
}

UnsqueezeOpx::UnsqueezeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<UnsqueezeOp>(op, {Onnx::Operators::Unsqueeze_1});
}

UnsqueezeGradOpx::UnsqueezeGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<UnsqueezeGradOp>(op, Onnx::GradOperators::UnsqueezeGrad);
}

void UnsqueezeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, inId(UnsqueezeGradOp::getInIndex()));
  outTensor =
      outTensor.reshape(outInfo(UnsqueezeOp::getOutIndex()).shape_szt());
  insert(outId(UnsqueezeGradOp::getOutIndex()), outTensor);
}

namespace {
OpxCreator<UnsqueezeOpx> unsqueezeOpxCreator(Onnx::Operators::Unsqueeze_1);
OpxCreator<UnsqueezeGradOpx>
    unsqueezeGradOpxCreator(Onnx::GradOperators::UnsqueezeGrad);
} // namespace

} // namespace popx
} // namespace poponnx
