// Copyright (c) 2021 Tensorscore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_TENSORS_HPP
#define POPART__INTERNAL_IR_BINDINGS_TENSORS_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

/**
 * Add bindings for `popart::Tensors` class to pybind module.
 **/
void bindTensors(py::module_ &m);

} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_TENSORS_HPP
