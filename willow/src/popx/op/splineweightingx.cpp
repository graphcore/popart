// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/SplineWeighting.hpp>

#include <popart/graphcoreoperators.hpp>
#include <popart/op/splineweighting.hpp>
#include <popart/popx/op/splineweightingx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SplineWeightingx::SplineWeightingx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<SplineWeightingOp>(op, {Onnx::CustomOperators::SplineWeighting});
}

bool SplineWeightingx::outputCreatedExternally(OutIndex index) const {
  return false;
}

void SplineWeightingx::grow(poplar::program::Sequence &prog) const {
  poplar::Tensor input  = getInTensor(SplineWeightingOp::inputIndex());
  poplar::Tensor weight = getInTensor(SplineWeightingOp::weightIndex());
  poplar::Tensor basis  = getInTensor(SplineWeightingOp::basisIndex());
  poplar::Tensor weightIndex =
      getInTensor(SplineWeightingOp::weightIndexIndex());
  auto &graph = this->graph();

  const auto output = popops::splineWeighting(
      graph, input, weight, basis, weightIndex, prog, "/SplineWeighting");
  setOutTensor(SplineWeightingOp::outputIndex(), output);
}

namespace {
OpxCreator<SplineWeightingx>
    splineweightingOpxCreator(Onnx::CustomOperators::SplineWeighting);
} // namespace

} // namespace popx
} // namespace popart
