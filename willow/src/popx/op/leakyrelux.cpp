// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iterator>
#include <memory>

#include <popart/error.hpp>
#include <popart/op/leakyrelu.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/leakyrelux.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

namespace pe = popops::expr;

poplar::Tensor
LeakyReluComputex::outplace(poplar::program::Sequence &prog,
                            snap::Graph &graph,
                            const poplar::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &debug_prefix) const {
  auto out_tensor = cloneNcopy(prog, graph, tensor, dnai);
  inplace(prog, graph, out_tensor, dnai, debug_prefix);
  return out_tensor;
}

void LeakyReluComputex::inplace(poplar::program::Sequence &prog,
                                snap::Graph &graph,
                                const poplar::Tensor &tensor,
                                const poplar::DebugNameAndId &dnai,
                                const std::string &debug_prefix) const {
  // x < 0.0f ? alpha * x : x
  auto expression = pe::Select(pe::Mul(pe::Const(getAlpha()), pe::_1),
                               pe::_1,
                               pe::Lt(pe::_1, pe::Const(0.0f)));

  popops::mapInPlace(
      graph.getPoplarGraph(), expression, {tensor}, prog, {dnai, debug_prefix});
}

float LeakyReluComputex::getAlphaFromLReluOp(Op *op) {
  auto lrelu = dynamic_cast<LeakyReluOp *>(op);
  if (lrelu == nullptr) {
    throw error("Not a valid LeakyReluOp : {}", op->str());
  }
  return lrelu->getAlpha();
}

float LeakyReluComputex::getAlphaFromLReluInplaceOp(Op *op) {
  auto lrelu = dynamic_cast<LeakyReluInplaceOp *>(op);
  if (lrelu == nullptr) {
    throw error("Not a valid LeakyReluInplaceOp : {}", op->str());
  }
  return lrelu->getAlpha();
}

LeakyReluOpx::LeakyReluOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          LeakyReluComputex::get(LeakyReluComputex::getAlphaFromLReluOp(op))) {
  verifyOp<LeakyReluOp>(
      op, {Onnx::Operators::LeakyRelu_1, Onnx::Operators::LeakyRelu_6});
}

LeakyReluInplaceOpx::LeakyReluInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          LeakyReluComputex::get(
              LeakyReluComputex::getAlphaFromLReluInplaceOp(op))) {
  verifyOp<LeakyReluInplaceOp>(op, Onnx::CustomOperators::LeakyReluInplace);
}

LeakyReluGradOpx::LeakyReluGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<LeakyReluGradOp>(op, Onnx::GradOperators::LeakyReluGrad);
}

void LeakyReluGradOpx::grow(poplar::program::Sequence &prog) const {
  auto &op = getOp<LeakyReluGradOp>();

  poplar::Tensor grad  = getInTensor(0);
  poplar::Tensor input = getInTensor(1);

  float alpha = op.getAlpha();

  // (grad * (x < 0.0f ? alpha : 1))
  pe::Mul expression = pe::Mul(pe::Select(pe::Const(alpha),
                                          pe::Const(1.0f),
                                          pe::Lt(pe::_2, pe::Const(0.0f))),
                               pe::_1);

  auto output = popops::map(graph().getPoplarGraph(),
                            expression,
                            {grad, input},
                            prog,
                            debugContext("leakyrelu_grad"));

  setOutTensor(0, output);
}

namespace {
OpxCreator<LeakyReluOpx> leakyReluOpxCreator({Onnx::Operators::LeakyRelu_1,
                                              Onnx::Operators::LeakyRelu_6});
OpxCreator<LeakyReluInplaceOpx>
    leakyReluInplaceOpxCreator(Onnx::CustomOperators::LeakyReluInplace);
OpxCreator<LeakyReluGradOpx>
    leakyReluGradOpxCreator(Onnx::GradOperators::LeakyReluGrad);
} // namespace

} // namespace popx
} // namespace popart
