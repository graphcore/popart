// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popops/Rearrange.hpp>
#include <popart/op/swish.hpp>
#include <popart/popx/op/swishx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

SwishOpx::SwishOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, SwishComputex::get()) {
  verifyOp<SwishOp>(op, Onnx::CustomOperators::Swish);
}

snap::Tensor SwishComputex::outplace(snap::program::Sequence &prog,
                                     snap::Graph &graph,
                                     const snap::Tensor &tensor,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &debug_prefix) const {
  auto out_tensor = cloneNcopy(prog, graph, tensor, dnai);
  inplace(prog, graph, out_tensor, dnai, debug_prefix);
  return out_tensor;
}

void SwishComputex::inplace(snap::program::Sequence &prog,
                            snap::Graph &graph,
                            const snap::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &debug_prefix) const {
  popnn::nonLinearityInPlace(graph.getPoplarGraph(),
                             popnn::NonLinearityType::SWISH,
                             tensor.getPoplarTensor(),
                             prog.getPoplarSequence(),
                             {dnai, debug_prefix});
}

SwishInplaceOpx::SwishInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, SwishComputex::get()) {
  verifyOp<SwishInplaceOp>(op, Onnx::CustomOperators::SwishInplace);
}

SwishGradOpx::SwishGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<SwishGradOp>(op, Onnx::CustomGradOperators::SwishGrad);
}

void SwishGradOpx::grow(snap::program::Sequence &prog) const {
  const auto grad =
      getInTensor(SwishGradOp::getGradInIndex()).getPoplarTensor();
  const auto input =
      getInTensor(SwishGradOp::getFwdArgInIndex()).getPoplarTensor();

  auto gradRearranged =
      popops::rearrange::regroupIfBeneficial(graph().getPoplarGraph(),
                                             grad,
                                             input,
                                             prog.getPoplarSequence(),
                                             debugContext("regroup"));

  auto output = popnn::nonLinearityInputGradient(graph().getPoplarGraph(),
                                                 popnn::NonLinearityType::SWISH,
                                                 input,
                                                 gradRearranged,
                                                 prog.getPoplarSequence(),
                                                 debugContext("swish_grad"));

  setOutTensor(SwishGradOp::getOutIndex(), snap::Tensor{output, graph()});
}

namespace {
OpxCreator<SwishOpx> swishOpxCreator(Onnx::CustomOperators::Swish);
OpxCreator<SwishInplaceOpx>
    swishxInplaceOpxCreator(Onnx::CustomOperators::SwishInplace);
OpxCreator<SwishGradOpx>
    swishGradOpxCreator(Onnx::CustomGradOperators::SwishGrad);
} // namespace

} // namespace popx
} // namespace popart
