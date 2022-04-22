// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPRITHMSHOSTTENSOR_HPP
#define GUARD_NEURALNET_POPRITHMSHOSTTENSOR_HPP

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/ndarray/dtype.hpp>

#include "popart/datatype.hpp"

namespace popart {
class Half;
class Tensor;

template <typename T> inline poprithms::ndarray::DType getPoprithmsDType() {
  return poprithms::ndarray::get<T>();
}
template <> inline poprithms::ndarray::DType getPoprithmsDType<Half>() {
  return poprithms::ndarray::DType::Float16;
}

poprithms::ndarray::DType getPoprithmsDType(popart::DataType t);

/**
 * Cast a popart::Tensor into a poprithms::compute::host::Tensor.
 *
 * If the popart::Tensor does not have data, then an error is thrown.
 */
poprithms::compute::host::Tensor
getPoprithmsComputeHostTensor(const popart::Tensor &);

} // namespace popart

#endif
