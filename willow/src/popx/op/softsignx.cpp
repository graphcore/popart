// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popart/op/softsign.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/softsignx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

SoftSignOpx::SoftSignOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          std::unique_ptr<EwuComputex>(new SoftSignComputex())) {
  verifyOp<SoftSignOp>(op, {Onnx::Operators::Softsign_1});
}

void SoftSignComputex::inplace(snap::program::Sequence &prog,
                               snap::Graph &graph,
                               const snap::Tensor &tensor,
                               const poplar::DebugNameAndId &dnai,
                               const std::string &debug_prefix) const {
  // Softsign definition: x/(1+abs(x))
  auto expr = pe::Divide(pe::_1, pe::Add(pe::Const(1.0f), pe::Abs(pe::_1)));

  popops::mapInPlace(graph.getPoplarGraph(),
                     expr,
                     {tensor.getPoplarTensor()},
                     prog.getPoplarSequence(),
                     {dnai, debug_prefix});
}

SoftSignInplaceOpx::SoftSignInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          std::unique_ptr<EwuComputex>(new SoftSignComputex())) {
  verifyOp<SoftSignInplaceOp>(op, Onnx::CustomOperators::SoftSignInplace);
}

SoftSignGradOpx::SoftSignGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<SoftSignGradOp>(op, Onnx::GradOperators::SoftSignGrad);
}

void SoftSignGradOpx::grow(snap::program::Sequence &prog) const {
  const auto grad_in   = getInTensor(SoftSignGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(SoftSignGradOp::getFwdArgInIndex());

  // The derivative of the softsign activation function is:
  // 1/((1 + abs(x))^2)
  //
  // Applying the elementwise chain rule gives:
  //
  // grad_out = grad_in /((1 + abs(x))^2)
  auto expr = pe::Divide(
      pe::_1,
      pe::Pow(pe::Add(pe::Const(1.0f), pe::Abs(pe::_2)), pe::Const(2.0f)));

  auto output =
      popops::map(graph().getPoplarGraph(),
                  expr,
                  {grad_in.getPoplarTensor(), fwd_input.getPoplarTensor()},
                  prog.getPoplarSequence(),
                  debugContext("softsign_grad"));

  setOutTensor(SoftSignGradOp::getOutIndex(), snap::Tensor{output, graph()});
}

namespace {
OpxCreator<SoftSignOpx> softsignOpxCreator({Onnx::Operators::Softsign_1});
OpxCreator<SoftSignInplaceOpx>
    softsignInplaceOpxCreator(Onnx::CustomOperators::SoftSignInplace);
OpxCreator<SoftSignGradOpx>
    softsignGradOpxCreator(Onnx::GradOperators::SoftSignGrad);
} // namespace

} // namespace popx
} // namespace popart
