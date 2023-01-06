// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <string>
#include <poplar/Tensor.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popops/Rearrange.hpp>
#include <popart/op/swish.hpp>
#include <popart/popx/op/swishx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
class Graph;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

SwishOpx::SwishOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, SwishComputex::get()) {
  verifyOp<SwishOp>(op, Onnx::CustomOperators::Swish);
}

poplar::Tensor SwishComputex::outplace(poplar::program::Sequence &prog,
                                       poplar::Graph &graph,
                                       const poplar::Tensor &tensor,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &debug_prefix) const {
  auto out_tensor = cloneNcopy(prog, graph, tensor, dnai);
  inplace(prog, graph, out_tensor, dnai, debug_prefix);
  return out_tensor;
}

void SwishComputex::inplace(poplar::program::Sequence &prog,
                            poplar::Graph &graph,
                            const poplar::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &debug_prefix) const {
  popnn::nonLinearityInPlace(graph,
                             popnn::NonLinearityType::SWISH,
                             tensor,
                             prog,
                             {dnai, debug_prefix});
}

SwishInplaceOpx::SwishInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, SwishComputex::get()) {
  verifyOp<SwishInplaceOp>(op, Onnx::CustomOperators::SwishInplace);
}

SwishGradOpx::SwishGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SwishGradOp>(op, Onnx::CustomGradOperators::SwishGrad);
}

void SwishGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto grad  = getInTensor(SwishGradOp::getGradInIndex());
  const auto input = getInTensor(SwishGradOp::getFwdArgInIndex());

  auto gradRearranged = popops::rearrange::regroupIfBeneficial(
      graph(), grad, input, prog, debugContext("regroup"));

  auto output = popnn::nonLinearityInputGradient(graph(),
                                                 popnn::NonLinearityType::SWISH,
                                                 input,
                                                 gradRearranged,
                                                 prog,
                                                 debugContext("swish_grad"));

  setOutTensor(SwishGradOp::getOutIndex(), output);
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
