// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <iterator>
#include <memory>
#include <popart/error.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/selu.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/selux.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

namespace {
template <typename T> T *get_as(Op *op) {
  auto x = dynamic_cast<T *>(op);
  if (!x) {
    throw error("Failed to cast {} in Selu", op->str());
  }
  return x;
}
} // namespace

SeluOpx::SeluOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          SeluComputex::get(get_as<SeluOp>(op)->getAlpha(),
                            get_as<SeluOp>(op)->getGamma())) {
  verifyOp<SeluOp>(op, {Onnx::Operators::Selu_1, Onnx::Operators::Selu_6});
}

void SeluComputex::inplace(poplar::program::Sequence &prog,
                           poplar::Graph &graph,
                           const poplar::Tensor &tensor,
                           const std::string &debug_prefix) const {
  //   The Selu definition is:
  // x > 0 ? gamma*alpha*(exp(x)-1) : gamma*x
  // We can rewrite it as a min max:
  // gamma*(min(alpha*(exp(x)-1), 0)+max(0, x))
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(std::make_unique<pe::Mul>(
      pe::Const(this->getAlpha()), pe::Sub(pe::Exp(pe::_1), pe::Const(1.0f))));
  exprs.push_back(std::make_unique<pe::Min>(pe::Const(0.0f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Add>(pe::Max(pe::Const(0.0f), pe::_1),
                                            *exprs.back()));
  exprs.push_back(
      std::make_unique<pe::Mul>(pe::Const(this->getGamma()), *exprs.back()));

  popops::mapInPlace(graph, *exprs.back(), {tensor}, prog, debug_prefix);
}

SeluInplaceOpx::SeluInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          SeluComputex::get(get_as<SeluInplaceOp>(op)->getAlpha(),
                            get_as<SeluInplaceOp>(op)->getGamma())) {
  verifyOp<SeluInplaceOp>(op, Onnx::CustomOperators::SeluInplace);
}

SeluGradOpx::SeluGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SeluGradOp>(op, Onnx::GradOperators::SeluGrad);
}

void SeluGradOpx::grow(poplar::program::Sequence &prog) const {
  auto op              = getOp<SeluGradOp>();
  const auto input     = getInTensor(SeluGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(SeluGradOp::getFwdArgInIndex());

  // We can write down the gradient of the Elu as two pieces:
  // theta(fwd_input < 0)*alpha*exp(fwd_input) + theta(fwd_input > 0)
  // The theta function is again written as theta(x) = (1+sign(x))/2
  // which is 1 when the argument is positive and 0 elsewhere,
  // except for 0.5 at 0, which does not matter here since the other
  // terms will be 0.

  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;

  // This is the theta(fwd_arg < 0)
  exprs.push_back(std::make_unique<pe::Divide>(
      pe::Sub(pe::Const(1.0f), pe::Signum(pe::_2)), pe::Const(2.0f)));
  // This is the gamma*alpha*exp(fwd_arg)
  exprs.push_back(std::make_unique<pe::Mul>(
      pe::Mul(pe::Const(op.getAlpha()), pe::Exp(pe::_2)), *exprs.back()));
  // This is the addition with the theta(fwd_input) to select when the input is
  // larger than 0
  exprs.push_back(std::make_unique<pe::Add>(
      pe::Divide(pe::Add(pe::Const(1.0f), pe::Signum(pe::_2)), pe::Const(2.0f)),
      *exprs.back()));
  // We multiply all the result by the scale gamma
  exprs.push_back(
      std::make_unique<pe::Mul>(pe::Const(op.getGamma()), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, *exprs.back()));

  auto output = popops::map(graph(),
                            *exprs.back(),
                            {input, fwd_input},
                            prog,
                            debugPrefix("selu_grad"));

  setOutTensor(SeluGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<SeluOpx> seluOpxCreator({Onnx::Operators::Selu_1,
                                    Onnx::Operators::Selu_6});
OpxCreator<SeluInplaceOpx>
    seluInplaceOpxCreator(Onnx::CustomOperators::SeluInplace);
OpxCreator<SeluGradOpx> seluGradOpxCreator(Onnx::GradOperators::SeluGrad);
} // namespace

} // namespace popx
} // namespace popart
