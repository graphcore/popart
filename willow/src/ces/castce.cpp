// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poprithmshosttensor.hpp>
#include <popart/ces/castce.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/cast.hpp>
#include <popart/tensor.hpp>

namespace popart {

ConstExprCast::ConstExprCast(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprCast::compute() {
  const auto in0 = getPoprithmsComputeHostTensor(*inTensor(0));
  return in0.to(getPoprithmsDType(outInfo0().dataType())).getNativeCharVector();
}

} // namespace popart
