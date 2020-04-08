// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NP_UTILS_HPP
#define GUARD_NEURALNET_NP_UTILS_HPP

#include <pybind11/numpy.h>

#include <popart/tensorinfo.hpp>
namespace popart {

std::map<std::string, DataType> initNpTypeMap() {
  std::map<std::string, DataType> M;
  // see tensorinfo.hpp for the complete list of
  // DataTypes (defined originally in ONNX)
  M["float16"] = DataType::FLOAT16;
  M["float32"] = DataType::FLOAT;
  M["uint8"]   = DataType::UINT8;
  M["uint16"]  = DataType::UINT16;
  M["uint32"]  = DataType::UINT32;
  M["uint64"]  = DataType::UINT64;
  M["int8"]    = DataType::INT8;
  M["int16"]   = DataType::INT16;
  M["int32"]   = DataType::INT32;
  M["int64"]   = DataType::INT64;
  M["bool"]    = DataType::BOOL;
  return M;
}

DataType getDataTypeFromNpType(std::string npType) {
  const static std::map<std::string, DataType> M = initNpTypeMap();
  auto found                                     = M.find(npType);
  if (found == M.end()) {
    throw error("No numpy type {} registered in map to DataType", npType);
  }
  return found->second;
}

TensorInfo getTensorInfo(pybind11::array npArr) {
  auto dtype      = npArr.dtype();
  auto typeString = pybind11::str(dtype);
  auto tRank      = npArr.ndim();
  std::vector<int64_t> shape;
  for (int i = 0; i < tRank; ++i) {
    shape.push_back(npArr.shape(i));
  }
  return TensorInfo(getDataTypeFromNpType(typeString), shape);
}

} // namespace popart

#endif
