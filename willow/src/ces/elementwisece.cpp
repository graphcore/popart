// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <vector>
#include <poprithmshosttensor.hpp>
#include <popart/ces/elementwisece.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/add.hpp>
#include <popart/tensor.hpp>

namespace popart {

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

ConstExprMod::ConstExprMod(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprMod::compute() {
  return (getPoprithmsComputeHostTensor(*inTensor(0)) %
          getPoprithmsComputeHostTensor(*inTensor(1)))
      .getNativeCharVector();
}

} // namespace popart
