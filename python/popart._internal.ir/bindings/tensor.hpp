// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_TENSOR_HPP
#define POPART__INTERNAL_IR_BINDINGS_TENSOR_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

/**
 * Add bindings for `popart::Tensor`, `popart::TensorType`,
 * `popart::VariableUpdateType`, `popart::TensorTypeInfo` classes to pybind
 * module.
 **/
void bindTensor(py::module &m);

} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_TENSOR_HPP
