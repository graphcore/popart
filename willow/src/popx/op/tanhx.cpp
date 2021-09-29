// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popnn/NonLinearity.hpp>
#include <popart/error.hpp>
#include <popart/op/tanh.hpp>
#include <popart/popx/op/tanhx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

TanhOpx::TanhOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<TanhOp>(op, {Onnx::Operators::Tanh_6});
}

void TanhOpx::grow(snap::program::Sequence &prog) const {
  auto in_tensor = getInTensor(TanhOp::getInIndex()).getPoplarTensor();
  // auto out_id     = outId(TanhOp::getOutIndex());
  auto out_tensor = popnn::nonLinearity(graph().getPoplarGraph(),
                                        popnn::NonLinearityType::TANH,
                                        in_tensor,
                                        prog.getPoplarSequence(),
                                        debugContext("nonLinearity"));
  setOutTensor(TanhOp::getOutIndex(), snap::Tensor{out_tensor, graph()});
}

InputCreatorType TanhOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

snap::Tensor
TanhOpx::unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

view::RegMap TanhOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

TanhGradOpx::TanhGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<TanhGradOp>(op, Onnx::GradOperators::TanhGrad);
}

void TanhGradOpx::grow(snap::program::Sequence &prog) const {
  auto fwd_out  = getInTensor(TanhGradOp::getFwdOutInIndex()).getPoplarTensor();
  auto grad_out = getInTensor(TanhGradOp::getGradInIndex()).getPoplarTensor();
  // auto out_id     = outId(TanhGradOp::getOutIndex());
  auto out_tensor = popnn::nonLinearityInputGradient(
      graph().getPoplarGraph(),
      popnn::NonLinearityType::TANH,
      fwd_out,
      grad_out,
      prog.getPoplarSequence(),
      debugContext("nonLinearityInputGradient"));
  setOutTensor(TanhGradOp::getOutIndex(), snap::Tensor{out_tensor, graph()});
}

namespace {
OpxCreator<TanhOpx> tanhOpxCreator(Onnx::Operators::Tanh_6);
OpxCreator<TanhGradOpx> tanhGradOpxCreator(Onnx::GradOperators::TanhGrad);
} // namespace

} // namespace popx
} // namespace popart
