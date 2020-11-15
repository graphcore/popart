// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poprithmshosttensor.hpp>
#include <popart/ces/gatherce.hpp>
#include <popart/op/gather.hpp>
#include <popart/tensor.hpp>

namespace popart {

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
