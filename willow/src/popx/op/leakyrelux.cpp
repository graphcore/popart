// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iterator>
#include <memory>

#include <popart/error.hpp>
#include <popart/op/leakyrelu.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/leakyrelux.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

namespace pe = popops::expr;

snap::Tensor
LeakyReluComputex::outplace(snap::program::Sequence &prog,
                            snap::Graph &graph,
                            const snap::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &debug_prefix) const {
  auto out_tensor = cloneNcopy(prog, graph, tensor, dnai);
  inplace(prog, graph, out_tensor, dnai, debug_prefix);
  return out_tensor;
}

void LeakyReluComputex::inplace(snap::program::Sequence &prog,
                                snap::Graph &graph,
                                const snap::Tensor &tensor,
                                const poplar::DebugNameAndId &dnai,
                                const std::string &debug_prefix) const {
  // x < 0.0f ? alpha * x : x
  auto expression = pe::Select(pe::Mul(pe::Const(getAlpha()), pe::_1),
                               pe::_1,
                               pe::Lt(pe::_1, pe::Const(0.0f)));

  popops::mapInPlace(graph.getPoplarGraph(),
                     expression,
                     {tensor.getPoplarTensor()},
                     prog.getPoplarSequence(),
                     {dnai, debug_prefix});
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

void LeakyReluGradOpx::grow(snap::program::Sequence &prog) const {
  auto &op = getOp<LeakyReluGradOp>();

  auto grad = getInTensor(LeakyReluGradOp::getGradInIndex()).getPoplarTensor();
  auto input =
      getInTensor(LeakyReluGradOp::getFwdArgInIndex()).getPoplarTensor();

  auto alpha = op.getAlpha();

  // x < 0.0f : alpha * grad : grad)
  // Expression is built with grad on each side of the Select so data type can
  // be inferred correctly. Was a problem with float16.
  auto expression = pe::Select(pe::Mul(pe::Const(alpha), pe::_1),
                               pe::_1,
                               pe::Lt(pe::_2, pe::Const(0.0f)));

  auto output = popops::map(graph().getPoplarGraph(),
                            expression,
                            {grad, input},
                            prog.getPoplarSequence(),
                            debugContext("leakyrelu_grad"));

  setOutTensor(0, snap::Tensor{output, graph()});
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
