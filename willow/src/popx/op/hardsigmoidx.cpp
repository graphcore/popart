// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <iterator>
#include <memory>
#include <popart/error.hpp>
#include <popart/op/hardsigmoid.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/hardsigmoidx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

namespace {
template <typename T> T *get_as(Op *op) {
  auto x = dynamic_cast<T *>(op);
  if (!x) {
    throw error("Failed to cast {} in HardSigmoid", op->str());
  }
  return x;
}
} // namespace

HardSigmoidOpx::HardSigmoidOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          HardSigmoidComputex::get(get_as<HardSigmoidOp>(op)->getAlpha(),
                                   get_as<HardSigmoidOp>(op)->getBeta())) {
  verifyOp<HardSigmoidOp>(
      op, {Onnx::Operators::HardSigmoid_1, Onnx::Operators::HardSigmoid_6});
}

void HardSigmoidComputex::inplace(poplar::program::Sequence &prog,
                                  snap::Graph &graph,
                                  const poplar::Tensor &tensor,
                                  const poplar::DebugNameAndId &dnai,
                                  const std::string &debug_prefix) const {
  //   Hardsigmoid definition: max(0, min(1, alpha*x+beta))
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;

  // These two linea compute the linear combination alpha*x+beta
  exprs.push_back(
      std::make_unique<pe::Mul>(pe::Const(this->getAlpha()), pe::_1));
  exprs.push_back(
      std::make_unique<pe::Add>(pe::Const(this->getBeta()), *exprs.back()));
  // These two lineas compute the max-min part
  exprs.push_back(std::make_unique<pe::Min>(pe::Const(1.0f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Max>(pe::Const(0.0f), *exprs.back()));

  popops::mapInPlace(graph.getPoplarGraph(),
                     *exprs.back(),
                     {tensor},
                     prog,
                     {dnai, debug_prefix});
}

HardSigmoidInplaceOpx::HardSigmoidInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          HardSigmoidComputex::get(
              get_as<HardSigmoidInplaceOp>(op)->getAlpha(),
              get_as<HardSigmoidInplaceOp>(op)->getBeta())) {
  verifyOp<HardSigmoidInplaceOp>(op, Onnx::CustomOperators::HardSigmoidInplace);
}

HardSigmoidGradOpx::HardSigmoidGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<HardSigmoidGradOp>(op, Onnx::GradOperators::HardSigmoidGrad);
}

void HardSigmoidGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto &op       = getOp<HardSigmoidGradOp>();
  const auto input     = getInTensor(HardSigmoidGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(HardSigmoidGradOp::getFwdArgInIndex());

  // The derivative of the Hardsigmoid activation function is:
  // 0 if x > 1-beta/alpha
  // 0 if x < -beta/alpha
  // alpha otherwise
  // we are going to write it this way:
  // (theta+)*(theta-)*alpha
  // where theta+ is a function that is equal to 1 if x < (1-beta)/alpha and 0
  // otherwise theta- is 1 if x < -beta/alpha and 0 otherwise Both these
  // functions are implemented as theta(arg) = (1+sign(arg))/2 so if arg is
  // positive this is 0 if negative is 1.

  // Theta_minus is theta(fwd_input+beta/alpha)
  std::vector<std::unique_ptr<popops::expr::Expr>> theta_minus;
  theta_minus.push_back(std::make_unique<pe::Signum>(
      pe::Add(pe::_2, pe::Const(op.getBeta() / op.getAlpha()))));
  theta_minus.push_back(std::make_unique<pe::Divide>(
      pe::Add(pe::Const(1.0f), *theta_minus.back()), pe::Const(2.0f)));

  // Theta_plus is theta((1-beta)/alpha-fwd_input)
  std::vector<std::unique_ptr<popops::expr::Expr>> theta_plus;
  theta_plus.push_back(std::make_unique<pe::Signum>(
      pe::Sub(pe::Const((1.0f - op.getBeta()) / op.getAlpha()), pe::_2)));
  theta_plus.push_back(std::make_unique<pe::Divide>(
      pe::Add(pe::Const(1.0f), *theta_plus.back()), pe::Const(2.0f)));

  // The gradient is then theta_plus*theta_minus*alpha
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(std::make_unique<pe::Mul>(
      pe::Const(op.getAlpha()),
      pe::Mul(*theta_plus.back(), *theta_minus.back())));
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, *exprs.back()));

  auto output = popops::map(graph().getPoplarGraph(),
                            *exprs.back(),
                            {input, fwd_input},
                            prog,
                            debugContext("hardsigmoid_grad"));

  setOutTensor(HardSigmoidGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<HardSigmoidOpx> hardsigmoidOpxCreator(
    {Onnx::Operators::HardSigmoid_1, Onnx::Operators::HardSigmoid_6});
OpxCreator<HardSigmoidInplaceOpx>
    hardsigmoidInplaceOpxCreator(Onnx::CustomOperators::HardSigmoidInplace);
OpxCreator<HardSigmoidGradOpx>
    hardsigmoidGradOpxCreator(Onnx::GradOperators::HardSigmoidGrad);
} // namespace

} // namespace popx
} // namespace popart
