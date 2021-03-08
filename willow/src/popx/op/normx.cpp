// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/normx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popnn/GroupNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace popart {
namespace popx {

// convert inverse standard deviation to variance
poplar::Tensor NormOpx::convertInvSdToVar(poplar::program::Sequence &prog,
                                          const poplar::Tensor &invSd,
                                          float epsilon,
                                          const poplar::Type dstType) const {

  return popops::invStdDevToVariance(
      graph(), invSd, epsilon, prog, dstType, debugContext("invSdToVar"));
}

// convert variant to inverse standard deviation
poplar::Tensor NormOpx::convertVarToInvSd(poplar::program::Sequence &prog,
                                          const poplar::Tensor &var,
                                          float epsilon,
                                          const poplar::Type dstType) const {
  // Investigation into the regression
  // https://phabricator.sourcevertex.net/T35286 has found that the call to
  // popops::map is faster. This should be investigated in poplibs, but while
  // the issue is present, the popops::map call should still be used if
  // possible.
  if (var.elementType() == dstType) {
    return popops::map(graph(),
                       pe::VarianceToInvStdDev(pe::_1, pe::Const(epsilon)),
                       {var},
                       prog,
                       debugContext("varToInvSd"));
  } else {
    return popops::varianceToInvStdDev(
        graph(), var, epsilon, prog, dstType, debugContext("varToInvSd"));
  }
}

NormOpx::NormOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

} // namespace popx
} // namespace popart
