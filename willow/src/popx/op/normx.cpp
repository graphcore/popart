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
snap::Tensor NormOpx::convertInvSdToVar(poplar::program::Sequence &prog,
                                        const snap::Tensor &invSd,
                                        float epsilon,
                                        const poplar::Type dstType) const {

  return snap::Tensor{popops::invStdDevToVariance(graph().getPoplarGraph(),
                                                  invSd.getPoplarTensor(),
                                                  epsilon,
                                                  prog,
                                                  dstType,
                                                  debugContext("invSdToVar")),
                      graph()};
}

// convert variant to inverse standard deviation
snap::Tensor NormOpx::convertVarToInvSd(poplar::program::Sequence &prog,
                                        const snap::Tensor &var,
                                        float epsilon,
                                        const poplar::Type dstType) const {

  return snap::Tensor{popops::varianceToInvStdDev(graph().getPoplarGraph(),
                                                  var.getPoplarTensor(),
                                                  epsilon,
                                                  prog,
                                                  dstType,
                                                  debugContext("varToInvSd")),
                      graph()};
}

NormOpx::NormOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

} // namespace popx
} // namespace popart
