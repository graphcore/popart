// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PYBIND11_ARRAY_ACCESSOR_HPP
#define GUARD_NEURALNET_PYBIND11_ARRAY_ACCESSOR_HPP

#include <pybind11/numpy.h>

namespace popart {
namespace StepIONS {
struct PyArrayAccessor {

  static void *getDataPointer(pybind11::array &array) {
    return array.request().ptr;
  }

  static size_t getArraySize(const pybind11::array &array) {
    return array.size();
  }

  static DataType getArrayDataType(pybind11::array &array) {
    auto dtype      = array.dtype();
    auto typeString = pybind11::str(dtype);
    return getDataTypeFromNpType(typeString);
  }

  static size_t getArrayRank(pybind11::array &array) { return array.ndim(); }

  static int64_t getArrayDim(pybind11::array &array, size_t index) {
    return array.shape(index);
  }
};

} // namespace StepIONS
} // namespace popart

#endif
