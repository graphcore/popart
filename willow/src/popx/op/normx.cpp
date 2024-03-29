// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <ext/new_allocator.h>
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/normx.hpp>

#include "popart/popx/opx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

namespace poplar {
namespace program {
class Sequence;
} // namespace program

using Shape = std::vector<std::size_t>;
} // namespace poplar

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

  return popops::varianceToInvStdDev(
      graph(), var, epsilon, prog, dstType, debugContext("varToInvSd"));
}

NormOpx::NormOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

} // namespace popx
} // namespace popart
