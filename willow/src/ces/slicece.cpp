// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <array>
#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/slicece.hpp>
#include <popart/op/slice.hpp>

#include "popart/ces/constexpr.hpp"

namespace popart {
class Op;

ConstExprSlice::ConstExprSlice(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprSlice::compute() {

  const auto lu = getOp<BaseSliceOp>().getLowerUpper();

  return getPoprithmsComputeHostTensor(*inTensor(0))
      .slice(std::get<0>(lu), std::get<1>(lu))
      .getNativeCharVector();
}

} // namespace popart
