// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/elementwisece.hpp>

#include "popart/ces/constexpr.hpp"

namespace popart {
class Op;

using namespace poprithms::compute;

ConstExprDiv::ConstExprDiv(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprDiv::compute() {
  return (getPoprithmsComputeHostTensor(*inTensor(0)) /
          getPoprithmsComputeHostTensor(*inTensor(1)))
      .getNativeCharVector();
}

ConstExprAdd::ConstExprAdd(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprAdd::compute() {
  return (getPoprithmsComputeHostTensor(*inTensor(0)) +
          getPoprithmsComputeHostTensor(*inTensor(1)))
      .getNativeCharVector();
}

ConstExprMul::ConstExprMul(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprMul::compute() {
  return (getPoprithmsComputeHostTensor(*inTensor(0)) *
          getPoprithmsComputeHostTensor(*inTensor(1)))
      .getNativeCharVector();
}

ConstExprSub::ConstExprSub(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprSub::compute() {
  return (getPoprithmsComputeHostTensor(*inTensor(0)) -
          getPoprithmsComputeHostTensor(*inTensor(1)))
      .getNativeCharVector();
}

ConstExprFmod::ConstExprFmod(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprFmod::compute() {
  return (getPoprithmsComputeHostTensor(*inTensor(0)) %
          getPoprithmsComputeHostTensor(*inTensor(1)))
      .getNativeCharVector();
}

} // namespace popart
