// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <onnx/onnx_pb.h>
#include <vector>
#include <poprithmshosttensor.hpp>
#include <popart/ces/slicece.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/slice.hpp>
#include <popart/tensor.hpp>

namespace popart {

ConstExprSlice::ConstExprSlice(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprSlice::compute() {
  auto upp = inShape(0);
  std::vector<int64_t> low(upp.size(), 0);
  for (auto slice : getOp<BaseSliceOp>().getSlices()) {
    low[slice.axis] = slice.start;
    upp[slice.axis] = slice.end;
  }

  return getPoprithmsComputeHostTensor(*inTensor(0))
      .slice(low, upp)
      .getNativeCharVector();
}

} // namespace popart
