// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_FLOAT8CONVERSION_HPP_
#define POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_FLOAT8CONVERSION_HPP_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

/**
 * Add bindings for `popart::Float8conversion` class to pybind module.
 **/
void bindFloat8conversion(py::module &m);

} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_FLOAT8CONVERSION_HPP_
