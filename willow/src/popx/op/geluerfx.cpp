// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <string>
#include <poplar/Tensor.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popops/Rearrange.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/op/geluerf.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/op/geluerfx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/opxmanager.hpp"

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

GeluErfOpx::GeluErfOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, GeluErfComputex::get()) {
  verifyOp<GeluErfOp>(op, {Onnx::CustomOperators::GeluErf_1});
}

poplar::Tensor
GeluErfComputex::outplace(poplar::program::Sequence &prog,
                          poplar::Graph &graph,
                          const poplar::Tensor &tensor,
                          const poplar::DebugNameAndId &dnai,
                          const std::string &debug_prefix) const {
  auto out_tensor = cloneNcopy(prog, graph, tensor, dnai);
  inplace(prog, graph, out_tensor, dnai, debug_prefix);
  return out_tensor;
}

void GeluErfComputex::inplace(poplar::program::Sequence &prog,
                              poplar::Graph &graph,
                              const poplar::Tensor &tensor,
                              const poplar::DebugNameAndId &dnai,
                              const std::string &debug_prefix) const {
  popnn::nonLinearityInPlace(graph,
                             popnn::NonLinearityType::GELU_ERF,
                             tensor,
                             prog,
                             {dnai, debug_prefix});
}

GeluErfInplaceOpx::GeluErfInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, GeluErfComputex::get()) {
  verifyOp<GeluErfInplaceOp>(op, Onnx::CustomOperators::GeluErfInplace);
}

GeluErfGradOpx::GeluErfGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GeluErfGradOp>(op, Onnx::GradOperators::GeluErfGrad);
}

void GeluErfGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto grad  = getInTensor(GeluErfGradOp::getGradInIndex());
  const auto input = getInTensor(GeluErfGradOp::getFwdArgInIndex());

  auto gradRearranged = popops::rearrange::regroupIfBeneficial(
      graph(), grad, input, prog, debugContext("regroup"));

  auto output =
      popnn::nonLinearityInputGradient(graph(),
                                       popnn::NonLinearityType::GELU_ERF,
                                       input,
                                       gradRearranged,
                                       prog,
                                       debugContext("geluerf_grad"));

  setOutTensor(GeluErfGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<GeluErfOpx> geluErfOpxCreator(Onnx::CustomOperators::GeluErf_1);
OpxCreator<GeluErfInplaceOpx>
    geluErfInplaceOpxCreator(Onnx::CustomOperators::GeluErfInplace);
OpxCreator<GeluErfGradOpx>
    geluErfGradOpxCreator(Onnx::GradOperators::GeluErfGrad);
} // namespace

} // namespace popx
} // namespace popart
