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
                                          float epsilon) const {
  return popops::map(graph(),
                     pe::InvStdDevToVariance(pe::_1, pe::Const(epsilon)),
                     {invSd},
                     prog,
                     debugContext("invSdToVar"));
}

// convert variant to inverse standard deviation
poplar::Tensor NormOpx::convertVarToInvSd(poplar::program::Sequence &prog,
                                          const poplar::Tensor &var,
                                          float epsilon) const {
  return popops::map(graph(),
                     pe::VarianceToInvStdDev(pe::_1, pe::Const(epsilon)),
                     {var},
                     prog,
                     debugContext("varToInvSd"));
}

NormOpx::NormOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

} // namespace popx
} // namespace popart
