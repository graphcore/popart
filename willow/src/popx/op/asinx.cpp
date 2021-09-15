// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <iterator>
#include <vector>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/asin.hpp>
#include <popart/popx/op/asinx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

AsinInplaceOpx::AsinInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, AsinComputex::get()) {
  verifyOp<AsinInplaceOp>(op, Onnx::CustomOperators::AsinInplace);
}

AsinOpx::AsinOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, AsinComputex::get()) {
  verifyOp<AsinOp>(op, Onnx::Operators::Asin_7);
}

snap::Tensor AsinComputex::outplace(poplar::program::Sequence &p,
                                    snap::Graph &g,
                                    const snap::Tensor &t,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t, dnai);
  inplace(p, g, outTensor, dnai, s);
  return outTensor;
}

void AsinComputex::inplace(poplar::program::Sequence &p,
                           snap::Graph &g,
                           const snap::Tensor &t,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  popops::mapInPlace(g.getPoplarGraph(),
                     popops::expr::UnaryOpType::ASIN,
                     t.getPoplarTensor(),
                     p,
                     {dnai, s});
}

AsinGradOpx::AsinGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<AsinGradOp>(op, Onnx::GradOperators::AsinGrad);
}

void AsinGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto input =
      getInTensor(AsinGradOp::getGradInIndex()).getPoplarTensor();
  const auto fwd_input =
      getInTensor(AsinGradOp::getFwdArgInIndex()).getPoplarTensor();

  // The derivative of the asin function can be constructed from normal
  // functions d/dx asin(x) = 1/sqrt(1-x^2)
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(
      std::make_unique<pe::Sub>(pe::Const(1.0f), pe::Mul(pe::_2, pe::_2)));
  exprs.push_back(std::make_unique<pe::Sqrt>(*exprs.back()));
  exprs.push_back(std::make_unique<pe::Divide>(pe::Const(1.0f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, *exprs.back()));

  auto output = popops::map(graph().getPoplarGraph(),
                            *exprs.back(),
                            {input, fwd_input},
                            prog,
                            debugContext("inverse_sine_grad"));

  setOutTensor(AsinGradOp::getOutIndex(), snap::Tensor{output, graph()});
}

namespace {
OpxCreator<AsinOpx> asinOpxCreator(Onnx::Operators::Asin_7);
OpxCreator<AsinInplaceOpx>
    asinInplaceOpxCreator(Onnx::CustomOperators::AsinInplace);
OpxCreator<AsinGradOpx> asinGradOpxCreator(Onnx::GradOperators::AsinGrad);
} // namespace

} // namespace popx
} // namespace popart
