// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NP_UTILS_HPP
#define GUARD_NEURALNET_NP_UTILS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <popart/tensorinfo.hpp>
#include <popart/vendored/optional.hpp>

// The following code allow nonstd::optional to be used in the C++
// interface and map to python types
namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<nonstd::optional<T>> : optional_caster<nonstd::optional<T>> {
};

} // namespace detail
} // namespace pybind11

namespace popart {

std::map<std::string, DataType> initNpTypeMap();

DataType getDataTypeFromNpType(std::string npType);

TensorInfo getTensorInfo(pybind11::array npArr);

// Check if npArr is c-contiguous in memory.
bool isContiguous(pybind11::array npArr);

// Check return an array with the same underlying data as npArr and is
// guaranteed to be c-contiguous.
pybind11::array makeContiguous(pybind11::array npArr);

} // namespace popart

#endif