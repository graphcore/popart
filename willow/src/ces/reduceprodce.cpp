// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/reduceprodce.hpp>
#include <popart/op/reduceprod.hpp>

#include "popart/ces/constexpr.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Op;

ConstExprReduceProd::ConstExprReduceProd(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprReduceProd::compute() {

  const auto tensor = getOp<ReduceProdOp>();

  const auto in0 =
      getPoprithmsComputeHostTensor(*inTensor(ReduceProdOp::getInIndex()));
  const auto in0_shape = in0.shape().get();

  // Fetch the axes and keepdims. Axes should be in the range [0, axes.size()-1]
  // after op.setup()
  const auto axes     = tensor.getAxes();
  const auto keepdims = tensor.getKeepDims();

  auto reducedTensor = in0.reduceProduct(tensor.backwardShape());

  if (!keepdims) { // remove the reduced axes if necessary
    reducedTensor = reducedTensor.reshape(outInfo0().shape());
  }

  return reducedTensor.getNativeCharVector();
}

} // namespace popart
