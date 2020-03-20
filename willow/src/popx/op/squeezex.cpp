// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/squeeze.hpp>
#include <popart/popx/op/squeezex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace popx {

SqueezeOpx::SqueezeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SqueezeOp>(
      op, {Onnx::Operators::Squeeze_1, Onnx::Operators::Squeeze_11});
}

void SqueezeOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, getInTensor(SqueezeOp::getInIndex()));
  outTensor = outTensor.reshape(outInfo(SqueezeOp::getOutIndex()).shape_szt());
  setOutTensor(SqueezeOp::getOutIndex(), outTensor);
}

SqueezeInplaceOpx::SqueezeInplaceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<SqueezeInplaceOp>(op, {Onnx::CustomOperators::SqueezeInplace});
}

void SqueezeInplaceOpx::grow(poplar::program::Sequence &) const {
  auto outTensor =
      getInTensor(SqueezeOp::getInIndex())
          .reshape(outInfo(SqueezeInplaceOp::getOutIndex()).shape_szt());
  setOutTensor(SqueezeOp::getOutIndex(), outTensor);
}

SqueezeGradOpx::SqueezeGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SqueezeGradOp>(op, Onnx::GradOperators::SqueezeGrad);
}

void SqueezeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, getInTensor(SqueezeGradOp::getInIndex()));
  outTensor = outTensor.reshape(outInfo(SqueezeOp::getOutIndex()).shape_szt());
  setOutTensor(SqueezeGradOp::getOutIndex(), outTensor);
}

namespace {
OpxCreator<SqueezeOpx> squeezeOpxCreator({Onnx::Operators::Squeeze_1,
                                          Onnx::Operators::Squeeze_11});
OpxCreator<SqueezeInplaceOpx>
    squeezeInplaceOpxCreator(Onnx::CustomOperators::SqueezeInplace);
OpxCreator<SqueezeGradOpx>
    squeezeGradOpxCreator(Onnx::GradOperators::SqueezeGrad);
} // namespace

} // namespace popx
} // namespace popart
