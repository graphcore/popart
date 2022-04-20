// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithmshosttensor.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordata.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

poprithms::ndarray::DType getPoprithmsDType(popart::DataType t) {
  switch (t) {
  case DataType::BOOL:
    return poprithms::ndarray::DType::Boolean;

  case DataType::UINT8:
    return poprithms::ndarray::DType::Unsigned8;

  case DataType::INT8:
    return poprithms::ndarray::DType::Int8;

  case DataType::UINT16:
    return poprithms::ndarray::DType::Unsigned16;

  case DataType::INT16:
    return poprithms::ndarray::DType::Int16;

  case DataType::UINT32:
    return poprithms::ndarray::DType::Unsigned32;

  case DataType::INT32:
    return poprithms::ndarray::DType::Int32;

  case DataType::UINT64:
    return poprithms::ndarray::DType::Unsigned64;

  case DataType::INT64:
    return poprithms::ndarray::DType::Int64;

  case DataType::FLOAT16:
    return poprithms::ndarray::DType::Float16;

  case DataType::FLOAT:
    return poprithms::ndarray::DType::Float32;

  case DataType::DOUBLE:
    return poprithms::ndarray::DType::Float64;

  default: {
    std::ostringstream oss;
    oss << "No poprithms::DType for popart::DataType " << t << '.';
    throw error(oss.str());
  }
  }
}

poprithms::compute::host::Tensor
getPoprithmsComputeHostTensor(const popart::Tensor &t) {
  return poprithms::compute::host::Tensor::copy(
      getPoprithmsDType(t.info.dataType()),
      t.info.shape(),
      t.tensorData()->data());
}

} // namespace popart
