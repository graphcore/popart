// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <iterator>
#include <memory>
#include <popart/error.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/thresholdedrelu.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/thresholdedrelux.hpp>
#include <popart/popx/opxmanager.hpp>

#include <snap/popops/ElementWise.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

namespace {
template <typename T> T *get_as(Op *op) {
  auto x = dynamic_cast<T *>(op);
  if (!x) {
    throw error("Failed to cast {} in Thresholded Relu", op->str());
  }
  return x;
}
} // namespace

ThresholdedReluOpx::ThresholdedReluOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          ThresholdedReluComputex::get(
              get_as<ThresholdedReluOp>(op)->getAlpha())) {
  verifyOp<ThresholdedReluOp>(op, {Onnx::Operators::ThresholdedRelu_10});
}

void ThresholdedReluComputex::inplace(snap::program::Sequence &prog,
                                      snap::Graph &graph,
                                      const snap::Tensor &tensor,
                                      const poplar::DebugNameAndId &dnai,
                                      const std::string &debug_prefix) const {

  // x < alpha ? 0 : x
  auto expression = pe::Select(
      pe::Const(0.0f), pe::_1, pe::Lte(pe::_1, pe::Const(getAlpha())));

  snap::popops::mapInPlace(
      graph, expression, {tensor}, prog, {dnai, debug_prefix});
}

ThresholdedReluInplaceOpx::ThresholdedReluInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          ThresholdedReluComputex::get(
              get_as<ThresholdedReluInplaceOp>(op)->getAlpha())) {
  verifyOp<ThresholdedReluInplaceOp>(
      op, Onnx::CustomOperators::ThresholdedReluInplace);
}

ThresholdedReluGradOpx::ThresholdedReluGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ThresholdedReluGradOp>(op, Onnx::GradOperators::ThresholdedReluGrad);
}

void ThresholdedReluGradOpx::grow(snap::program::Sequence &prog) const {
  const auto &op       = getOp<ThresholdedReluGradOp>();
  const auto input     = getInTensor(ThresholdedReluGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(ThresholdedReluGradOp::getFwdArgInIndex());

  // x <= alpha ? 0 : 1
  auto expression =
      pe::Mul(pe::Select(pe::Const(0.0f),
                         pe::Const(1.0f),
                         pe::Lte(pe::_2, pe::Const(op.getAlpha()))),
              pe::_1);

  auto output = snap::popops::map(graph(),
                                  expression,
                                  {input, fwd_input},
                                  prog,
                                  debugContext("thresholdedrelu_grad"));

  setOutTensor(ThresholdedReluGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ThresholdedReluOpx>
    thresholdedreluOpxCreator({Onnx::Operators::ThresholdedRelu_10});
OpxCreator<ThresholdedReluInplaceOpx> thresholdedreluInplaceOpxCreator(
    Onnx::CustomOperators::ThresholdedReluInplace);
OpxCreator<ThresholdedReluGradOpx>
    thresholdedreluGradOpxCreator(Onnx::GradOperators::ThresholdedReluGrad);
} // namespace

} // namespace popx
} // namespace popart
