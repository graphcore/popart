// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_PATTERNS_PATTERNS_HPP_
#define POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_PATTERNS_PATTERNS_HPP_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace patterns {

/**
 * Add bindings for `popart::Patterns` class to pybind module.
 **/
void bindPatterns(py::module &m);

} // namespace patterns
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_PATTERNS_PATTERNS_HPP_
