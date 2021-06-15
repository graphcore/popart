// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <onnxutil.hpp>
#include <poprithms/compute/host/usings.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/scalece.hpp>
#include <popart/op/scale.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::vector<char> ConstExprScale::compute() {
  auto f = poprithms::compute::host::Tensor::scalar(
      poprithms::compute::host::DType::Float64,
      static_cast<double>(getOp<ScaleOp>().getScaleFactor()));

  return (getPoprithmsComputeHostTensor(*inTensor(0)).toFloat64().mul(f))
      .to(getPoprithmsDType(inTensor(0)->info.dataType()))
      .getNativeCharVector();
}

} // namespace popart
