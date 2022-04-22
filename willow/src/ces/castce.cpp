// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/castce.hpp>

#include "popart/ces/constexpr.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Op;

ConstExprCast::ConstExprCast(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprCast::compute() {
  const auto in0 = getPoprithmsComputeHostTensor(*inTensor(0));
  return in0.to(getPoprithmsDType(outInfo0().dataType())).getNativeCharVector();
}

} // namespace popart
