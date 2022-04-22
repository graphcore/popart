// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/floorce.hpp>

#include "popart/ces/constexpr.hpp"

namespace popart {
class Op;

ConstExprFloor::ConstExprFloor(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprFloor::compute() {
  return getPoprithmsComputeHostTensor(*inTensor(0))
      .floor()
      .getNativeCharVector();
}

} // namespace popart
