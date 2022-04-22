// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/softsign.hpp>
#include <popart/popx/op/softsignx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

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

  snap::popops::mapInPlace(graph, expr, {tensor}, prog, {dnai, debug_prefix});
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

  auto output = snap::popops::map(
      graph(), expr, {grad_in, fwd_input}, prog, debugContext("softsign_grad"));

  setOutTensor(SoftSignGradOp::getOutIndex(), output);
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
