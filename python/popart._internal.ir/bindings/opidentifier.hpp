// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OPIDENTIFIER_HPP_
#define POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OPIDENTIFIER_HPP_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

/**
 * Add bindings for `popart::OpIdentifier` class to pybind module.
 **/
void bindOpIdentifier(py::module &m);

} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OPIDENTIFIER_HPP_
