// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/gatherce.hpp>
#include <popart/op/gather.hpp>

#include "popart/ces/constexpr.hpp"

namespace popart {
class Op;

ConstExprGather::ConstExprGather(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprGather::compute() {
  return getPoprithmsComputeHostTensor(*inTensor(GatherOp::dataInIndex()))
      .gather(
          getOp<GatherOp>().getAxis(),
          getPoprithmsComputeHostTensor(*inTensor(GatherOp::indicesInIndex()))
              .getInt64Vector())
      .getNativeCharVector();
}

} // namespace popart
