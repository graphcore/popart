// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/concatce.hpp>
#include <popart/op/concat.hpp>
#include <popart/tensorindex.hpp>

#include "popart/ces/constexpr.hpp"
#include "popart/op.hpp"

namespace popart {

ConstExprConcat::ConstExprConcat(Op *op_) : ConstExprOp(op_) {
  axis        = getOp<ConcatOp>().getAxis();
  input_count = op_->input->n();
}

std::vector<char> ConstExprConcat::compute() {
  std::vector<poprithms::compute::host::Tensor> rithms;
  rithms.reserve(input_count);
  for (int i = 0; i < input_count; i++) {
    rithms.push_back(getPoprithmsComputeHostTensor(*inTensor(i)));
  }

  return poprithms::compute::host::Tensor::concat(rithms, axis)
      .getNativeCharVector();
}

} // namespace popart
