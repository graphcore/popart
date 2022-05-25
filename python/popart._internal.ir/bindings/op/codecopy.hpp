// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_CODECOPY_HPP
#define POPART__INTERNAL_IR_BINDINGS_CODECOPY_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {
namespace exchange {

/**
 * Add bindings for the codecopy op.
 **/
void bindCodeCopy(py::module &m);

} // namespace exchange
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_CODECOPY_HPP
