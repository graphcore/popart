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

  return popops::invStdDevToVariance(graph().getPoplarGraph(),
                                     invSd,
                                     epsilon,
                                     prog,
                                     dstType,
                                     debugContext("invSdToVar"));
}

// convert variant to inverse standard deviation
poplar::Tensor NormOpx::convertVarToInvSd(poplar::program::Sequence &prog,
                                          const poplar::Tensor &var,
                                          float epsilon,
                                          const poplar::Type dstType) const {

  return popops::varianceToInvStdDev(graph().getPoplarGraph(),
                                     var,
                                     epsilon,
                                     prog,
                                     dstType,
                                     debugContext("varToInvSd"));
}

NormOpx::NormOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

} // namespace popx
} // namespace popart
