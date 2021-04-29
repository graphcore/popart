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

  const auto lu = getOp<BaseSliceOp>().getLowerUpper();

  return getPoprithmsComputeHostTensor(*inTensor(0))
      .slice(std::get<0>(lu), std::get<1>(lu))
      .getNativeCharVector();
}

} // namespace popart
