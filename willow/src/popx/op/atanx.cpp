// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <iterator>
#include <vector>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/atan.hpp>
#include <popart/popx/op/atanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

AtanInplaceOpx::AtanInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, AtanComputex::get()) {
  verifyOp<AtanInplaceOp>(op, Onnx::CustomOperators::AtanInplace);
}

AtanOpx::AtanOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, AtanComputex::get()) {
  verifyOp<AtanOp>(op, Onnx::Operators::Atan_7);
}

snap::Tensor AtanComputex::outplace(poplar::program::Sequence &p,
                                    snap::Graph &g,
                                    const snap::Tensor &t,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t, dnai);
  inplace(p, g, outTensor, dnai, s);
  return outTensor;
}

void AtanComputex::inplace(poplar::program::Sequence &p,
                           snap::Graph &g,
                           const snap::Tensor &t,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  //   The formula for atan in the poplar device is giving problems, mapping not
  //   to the correct interval I rewrote it using the asin function which has
  //   proven to be working correctly
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(
      std::make_unique<pe::Add>(pe::Const(1.0f), pe::Mul(pe::_1, pe::_1)));
  exprs.push_back(std::make_unique<pe::Sqrt>(*exprs.back()));
  exprs.push_back(std::make_unique<pe::Divide>(pe::_1, *exprs.back()));
  exprs.push_back(std::make_unique<pe::Asin>(*exprs.back()));

  popops::mapInPlace(
      g.getPoplarGraph(), *exprs.back(), {t.getPoplarTensor()}, p, {dnai, s});
}

AtanGradOpx::AtanGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<AtanGradOp>(op, Onnx::GradOperators::AtanGrad);
}

void AtanGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto input =
      getInTensor(AtanGradOp::getGradInIndex()).getPoplarTensor();
  const auto fwd_input =
      getInTensor(AtanGradOp::getFwdArgInIndex()).getPoplarTensor();

  // The derivative of the atan function can be constructed from normal
  // functions d/dx atan(x) = 1/(1+x^2)
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(
      std::make_unique<pe::Add>(pe::Const(1.0f), pe::Mul(pe::_2, pe::_2)));
  exprs.push_back(std::make_unique<pe::Divide>(pe::Const(1.0f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, *exprs.back()));

  auto output = popops::map(graph().getPoplarGraph(),
                            *exprs.back(),
                            {input, fwd_input},
                            prog,
                            debugContext("inverse_tangent_grad"));

  setOutTensor(AtanGradOp::getOutIndex(), snap::Tensor{output, graph()});
}

namespace {
OpxCreator<AtanOpx> atanOpxCreator(Onnx::Operators::Atan_7);
OpxCreator<AtanInplaceOpx>
    atanInplaceOpxCreator(Onnx::CustomOperators::AtanInplace);
OpxCreator<AtanGradOpx> atanGradOpxCreator(Onnx::GradOperators::AtanGrad);
} // namespace

} // namespace popx
} // namespace popart
