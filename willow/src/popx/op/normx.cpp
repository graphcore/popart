// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/normx.hpp>

#include "popart/popx/popopx.hpp"

namespace popart {
class Op;
namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace popart {
namespace popx {

// convert inverse standard deviation to variance
snap::Tensor NormOpx::convertInvSdToVar(snap::program::Sequence &prog,
                                        const snap::Tensor &invSd,
                                        float epsilon,
                                        const poplar::Type dstType) const {

  return snap::Tensor{popops::invStdDevToVariance(graph().getPoplarGraph(),
                                                  invSd.getPoplarTensor(),
                                                  epsilon,
                                                  prog.getPoplarSequence(),
                                                  dstType,
                                                  debugContext("invSdToVar")),
                      graph()};
}

// convert variant to inverse standard deviation
snap::Tensor NormOpx::convertVarToInvSd(snap::program::Sequence &prog,
                                        const snap::Tensor &var,
                                        float epsilon,
                                        const poplar::Type dstType) const {

  return snap::Tensor{popops::varianceToInvStdDev(graph().getPoplarGraph(),
                                                  var.getPoplarTensor(),
                                                  epsilon,
                                                  prog.getPoplarSequence(),
                                                  dstType,
                                                  debugContext("varToInvSd")),
                      graph()};
}

NormOpx::NormOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

} // namespace popx
} // namespace popart
