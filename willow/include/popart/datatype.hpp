// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_DATATYPE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_DATATYPE_HPP_
#include <popart/basicoptionals.hpp>

namespace popart {
/// There is a one-to-one correspondence
/// between \c popart::DataTypes and
/// \c ONNX_NAMESPACE::TensorProto_DataTypes, which is equivalent to
/// \c decltype(ONNX_NAMESPACE::TensorProto().data_type()).
enum class DataType {
  // fixed point types
  UINT8 = 0,
  INT8,
  UINT16,
  INT16,
  INT32,
  INT64,
  UINT32,
  UINT64,
  BOOL,
  // floating point types
  FLOAT,
  FLOAT16,
  BFLOAT16,
  DOUBLE,
  COMPLEX64,
  COMPLEX128,
  // other types
  STRING,
  UNDEFINED,
};

using OptionalDataType = BasicOptional<DataType, 0>;

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_DATATYPE_HPP_
