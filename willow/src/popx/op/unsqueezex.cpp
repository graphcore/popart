#include <popart/error.hpp>
#include <popart/op/unsqueeze.hpp>
#include <popart/popx/op/unsqueezex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace popx {

void UnsqueezeOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, getInTensor(UnsqueezeOp::getInIndex()));
  outTensor =
      outTensor.reshape(outInfo(UnsqueezeOp::getOutIndex()).shape_szt());
  setOutTensor(UnsqueezeOp::getOutIndex(), outTensor);
}

UnsqueezeOpx::UnsqueezeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<UnsqueezeOp>(op, {Onnx::Operators::Unsqueeze_1});
}

UnsqueezeGradOpx::UnsqueezeGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<UnsqueezeGradOp>(op, Onnx::GradOperators::UnsqueezeGrad);
}

void UnsqueezeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, getInTensor(UnsqueezeGradOp::getInIndex()));
  outTensor =
      outTensor.reshape(outInfo(UnsqueezeOp::getOutIndex()).shape_szt());
  setOutTensor(UnsqueezeGradOp::getOutIndex(), outTensor);
}

namespace {
OpxCreator<UnsqueezeOpx> unsqueezeOpxCreator(Onnx::Operators::Unsqueeze_1);
OpxCreator<UnsqueezeGradOpx>
    unsqueezeGradOpxCreator(Onnx::GradOperators::UnsqueezeGrad);
} // namespace

} // namespace popx
} // namespace popart
