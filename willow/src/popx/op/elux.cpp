// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <vector>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/error.hpp>
#include <popart/op/elu.hpp>
#include <popart/popx/op/elux.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
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
namespace popx {
class Devicex;

namespace {
template <typename T> T *get_as(Op *op) {
  auto x = dynamic_cast<T *>(op);
  if (!x) {
    throw error("Failed to cast {} in Elu", op->str());
  }
  return x;
}
} // namespace

EluOpx::EluOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          EluComputex::get(get_as<EluOp>(op)->alpha())) {
  verifyOp<EluOp>(op, {Onnx::Operators::Elu_1, Onnx::Operators::Elu_6});
}

void EluComputex::inplace(snap::program::Sequence &prog,
                          snap::Graph &graph,
                          const snap::Tensor &tensor,
                          const poplar::DebugNameAndId &dnai,
                          const std::string &debug_prefix) const {

  //   The Elu definition is:
  // x > 0 ? alpha*(exp(x)-1) : x
  // We can rewrite it as:
  // (min(alpha*(exp(x)-1), 0)+max(0, x))
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(std::make_unique<pe::Mul>(
      pe::Const(this->alpha()), pe::Sub(pe::Exp(pe::_1), pe::Const(1.0f))));
  exprs.push_back(std::make_unique<pe::Min>(pe::Const(0.0f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Add>(pe::Max(pe::Const(0.0f), pe::_1),
                                            *exprs.back()));

  snap::popops::mapInPlace(
      graph, *exprs.back(), {tensor}, prog, {dnai, debug_prefix});
}

EluInplaceOpx::EluInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          EluComputex::get(get_as<EluInplaceOp>(op)->alpha())) {
  verifyOp<EluInplaceOp>(op, Onnx::CustomOperators::EluInplace);
}

EluGradOpx::EluGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<EluGradOp>(op, Onnx::GradOperators::EluGrad);
}

void EluGradOpx::grow(snap::program::Sequence &prog) const {
  const auto &op       = getOp<EluGradOp>();
  const auto input     = getInTensor(EluGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(EluGradOp::getFwdArgInIndex());

  // We can write down the gradient of the Elu as two pieces:
  // theta(fwd_input < 0)*alpha*exp(fwd_input) + theta(fwd_input > 0)
  // The theta function is again written as theta(x) = (1+sign(x))/2
  // which is 1 when the argument is positive and 0 elsewhere,
  // except for 0.5 at 0, which does not matter here since the other
  // terms will be 0.

  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;

  exprs.push_back(std::make_unique<pe::Divide>(
      pe::Sub(pe::Const(1.0f), pe::Signum(pe::_2)), pe::Const(2.0f)));
  exprs.push_back(std::make_unique<pe::Mul>(
      pe::Mul(pe::Const(op.alpha()), pe::Exp(pe::_2)), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Add>(
      pe::Divide(pe::Add(pe::Const(1.0f), pe::Signum(pe::_2)), pe::Const(2.0f)),
      *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, *exprs.back()));

  auto output = snap::popops::map(graph(),
                                  *exprs.back(),
                                  {input, fwd_input},
                                  prog,
                                  debugContext("elu_grad"));

  setOutTensor(EluGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<EluOpx> eluOpxCreator({Onnx::Operators::Elu_1,
                                  Onnx::Operators::Elu_6});
OpxCreator<EluInplaceOpx>
    eluInplaceOpxCreator(Onnx::CustomOperators::EluInplace);
OpxCreator<EluGradOpx> eluGradOpxCreator(Onnx::GradOperators::EluGrad);
} // namespace

} // namespace popx
} // namespace popart
