// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popart/op/softplus.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/softplusx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

SoftPlusOpx::SoftPlusOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          std::unique_ptr<EwuComputex>(new SoftPlusComputex())) {
  verifyOp<SoftPlusOp>(op, {Onnx::Operators::Softplus_1});
}

void SoftPlusComputex::inplace(snap::program::Sequence &prog,
                               snap::Graph &graph,
                               const snap::Tensor &tensor,
                               const poplar::DebugNameAndId &dnai,
                               const std::string &debug_prefix) const {
  // Softplus definition: ln(exp(x)+1)
  // This is equivalent to max(x,0) + ln(1+exp(-abs(x))), which is more
  // numerical stable
  auto expr = pe::Add(
      pe::Max(pe::_1, pe::Const(0.0f)),
      pe::Log(pe::Add(pe::Const(1.0f), pe::Exp(pe::Neg(pe::Abs(pe::_1))))));

  popops::mapInPlace(graph.getPoplarGraph(),
                     expr,
                     {tensor.getPoplarTensor()},
                     prog.getPoplarSequence(),
                     {dnai, debug_prefix});
}

SoftPlusInplaceOpx::SoftPlusInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          std::unique_ptr<EwuComputex>(new SoftPlusComputex())) {
  verifyOp<SoftPlusInplaceOp>(op, Onnx::CustomOperators::SoftPlusInplace);
}

SoftPlusGradOpx::SoftPlusGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<SoftPlusGradOp>(op, Onnx::GradOperators::SoftPlusGrad);
}

void SoftPlusGradOpx::grow(snap::program::Sequence &prog) const {
  const auto grad_in   = getInTensor(SoftPlusGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(SoftPlusGradOp::getFwdArgInIndex());

  // The derivative of the softplus activation function is:
  //
  // exp(x)/(exp(x) + 1) = 1/(exp(-x) + 1) = sigmoid(x)
  //
  // Applying the elementwise chain rule gives:
  //
  // grad_out = grad_in * sigmoid(x)
  auto output =
      popops::map(graph().getPoplarGraph(),
                  pe::_1 * pe::Sigmoid(pe::_2),
                  {grad_in.getPoplarTensor(), fwd_input.getPoplarTensor()},
                  prog.getPoplarSequence(),
                  debugContext("softplus_grad"));

  setOutTensor(SoftPlusGradOp::getOutIndex(), snap::Tensor{output, graph()});
}

namespace {
OpxCreator<SoftPlusOpx> softplusOpxCreator({Onnx::Operators::Softplus_1});
OpxCreator<SoftPlusInplaceOpx>
    softplusInplaceOpxCreator(Onnx::CustomOperators::SoftPlusInplace);
OpxCreator<SoftPlusGradOpx>
    softplusGradOpxCreator(Onnx::GradOperators::SoftPlusGrad);
} // namespace

} // namespace popx
} // namespace popart
