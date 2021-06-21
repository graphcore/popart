// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PYBIND11_ARRAY_ACCESSOR_HPP
#define GUARD_NEURALNET_PYBIND11_ARRAY_ACCESSOR_HPP

#include <pybind11/numpy.h>

namespace popart {
namespace StepIONS {

/* pybind11::array doesn't like multi-threading:
 * https://github.com/pybind/pybind11/issues/1723 so we wrap them by extracting
 * the information we need in the main thread, and don't actually pass the
 * pybind11 object to the worker threads.
 */
struct PyArray {
  PyArray(pybind11::array array) : _container(array) {
    ptr        = array.request().ptr;
    size       = array.size();
    auto dtype = array.dtype();
    // Store the dtype as string because in some cases it's not initialised
    // so delay the conversion to PopART DataType to when it's actually
    // requested.
    typeString = pybind11::str(dtype);
    shape.reserve(array.ndim());
    for (int i = 0; i < array.ndim(); ++i) {
      shape.push_back(array.shape(i));
    }
  }
  void *ptr;
  size_t size;
  std::string typeString;
  std::vector<int64_t> shape;

private:
  pybind11::array _container; // Used for storage only, don't access it outside
                              // the constructor (Will crash the application if
                              // accessed outside the main thread)
};

struct PyArrayAccessor {

  static void *getDataPointer(PyArray &array) { return array.ptr; }

  static size_t getArraySize(const PyArray &array) { return array.size; }

  static DataType getArrayDataType(PyArray &array) {
    return getDataTypeFromNpType(array.typeString);
  }

  static size_t getArrayRank(PyArray &array) { return array.shape.size(); }

  static int64_t getArrayDim(PyArray &array, size_t index) {
    return array.shape.at(index);
  }
};

} // namespace StepIONS
} // namespace popart

#endif
