// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <onnxutil.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/floorce.hpp>
#include <popart/op/floor.hpp>
#include <popart/tensor.hpp>

namespace popart {

ConstExprFloor::ConstExprFloor(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprFloor::compute() {
  return getPoprithmsComputeHostTensor(*inTensor(0))
      .floor()
      .getNativeCharVector();
}

} // namespace popart
