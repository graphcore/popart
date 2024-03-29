// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/sinh.hpp>
#include <popart/popx/op/sinhx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

SinhInplaceOpx::SinhInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, SinhComputex::get()) {
  verifyOp<SinhInplaceOp>(op, Onnx::CustomOperators::SinhInplace);
}

SinhOpx::SinhOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, SinhComputex::get()) {
  verifyOp<SinhOp>(op, Onnx::Operators::Sinh_9);
}

poplar::Tensor SinhComputex::outplace(poplar::program::Sequence &p,
                                      poplar::Graph &g,
                                      const poplar::Tensor &t,
                                      const poplar::DebugNameAndId &dnai,
                                      const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t, dnai);
  inplace(p, g, outTensor, dnai, s);
  return outTensor;
}

void SinhComputex::inplace(poplar::program::Sequence &p,
                           poplar::Graph &g,
                           const poplar::Tensor &t,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(
      std::make_unique<pe::Divide>(pe::Const(1.0f), pe::Exp(pe::_1)));
  exprs.push_back(std::make_unique<pe::Sub>(pe::Exp(pe::_1), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::Const(0.5f), *exprs.back()));

  // apply the inplace SINH
  popops::mapInPlace(g, *exprs.back(), {t}, p, {dnai, s});
}

SinhGradOpx::SinhGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SinhGradOp>(op, Onnx::GradOperators::SinhGrad);
}

void SinhGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto input     = getInTensor(SinhGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(SinhGradOp::getFwdArgInIndex());

  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(
      std::make_unique<pe::Divide>(pe::Const(1.0f), pe::Exp(pe::_2)));
  exprs.push_back(std::make_unique<pe::Add>(pe::Exp(pe::_2), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::Const(0.5f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, *exprs.back()));

  auto output = popops::map(graph(),
                            *exprs.back(),
                            {input, fwd_input},
                            prog,
                            debugContext("output_grad"));

  setOutTensor(SinhGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<SinhOpx> sinhOpxCreator(Onnx::Operators::Sinh_9);
OpxCreator<SinhInplaceOpx>
    sinhInplaceOpxCreator(Onnx::CustomOperators::SinhInplace);
OpxCreator<SinhGradOpx> sinhGradOpxCreator(Onnx::GradOperators::SinhGrad);
} // namespace

} // namespace popx
} // namespace popart
