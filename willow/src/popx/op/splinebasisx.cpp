// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/SplineBasis.hpp>

#include <popart/graphcoreoperators.hpp>
#include <popart/op/splinebasis.hpp>
#include <popart/popx/op/splinebasisx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

using namespace popops::expr;

SplineBasisx::SplineBasisx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SplineBasisOp>(op, {Onnx::CustomOperators::SplineBasis});
}

bool SplineBasisx::outputCreatedExternally(OutIndex index) const {
  return true;
}

void SplineBasisx::grow(poplar::program::Sequence &prog) const {
  const auto &srop      = getOp<SplineBasisOp>();
  const unsigned degree = srop.getDegree();

  poplar::Tensor pseudo       = getInTensor(SplineBasisOp::pseudoIndex());
  poplar::Tensor kernelSize   = getInTensor(SplineBasisOp::kernelSizeIndex());
  poplar::Tensor isOpenSpline = getInTensor(SplineBasisOp::isOpenSplineIndex());
  auto basis                  = getOutTensor(SplineBasisOp::outBasisIndex());
  auto weightIndex = getOutTensor(SplineBasisOp::outWeightIndexIndex());
  auto &graph      = this->graph();

  popops::splineBasis(graph,
                      pseudo,
                      kernelSize,
                      isOpenSpline,
                      basis,
                      weightIndex,
                      degree,
                      prog,
                      "/SplineBasis");
}

namespace {
OpxCreator<SplineBasisx>
    splinebasisOpxCreator(Onnx::CustomOperators::SplineBasis);
} // namespace

} // namespace popx
} // namespace popart
