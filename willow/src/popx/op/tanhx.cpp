// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popnn/NonLinearity.hpp>
#include <popart/error.hpp>
#include <popart/op/tanh.hpp>
#include <popart/popx/op/tanhx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

TanhOpx::TanhOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<TanhOp>(op, {Onnx::Operators::Tanh_6});
}

void TanhOpx::grow(poplar::program::Sequence &prog) const {
  auto in_tensor = getInTensor(TanhOp::getInIndex());
  // auto out_id     = outId(TanhOp::getOutIndex());
  auto out_tensor = popnn::nonLinearity(graph(),
                                        popnn::NonLinearityType::TANH,
                                        in_tensor,
                                        prog,
                                        debugPrefix("nonLinearity"));
  setOutTensor(TanhOp::getOutIndex(), out_tensor);
}

InputCreatorType TanhOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

poplar::Tensor
TanhOpx::unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

view::RegMap TanhOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

TanhGradOpx::TanhGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<TanhGradOp>(op, Onnx::GradOperators::TanhGrad);
}

void TanhGradOpx::grow(poplar::program::Sequence &prog) const {
  auto fwd_out  = getInTensor(TanhGradOp::getFwdOutInIndex());
  auto grad_out = getInTensor(TanhGradOp::getGradInIndex());
  // auto out_id     = outId(TanhGradOp::getOutIndex());
  auto out_tensor = popnn::nonLinearityInputGradient(
      graph(),
      popnn::NonLinearityType::TANH,
      fwd_out,
      grad_out,
      prog,
      debugPrefix("nonLinearityInputGradient"));
  setOutTensor(TanhGradOp::getOutIndex(), out_tensor);
}

namespace {
OpxCreator<TanhOpx> tanhOpxCreator(Onnx::Operators::Tanh_6);
OpxCreator<TanhGradOpx> tanhGradOpxCreator(Onnx::GradOperators::TanhGrad);
} // namespace

} // namespace popx
} // namespace popart
