// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_ROIALIGN_HPP
#define POPART__INTERNAL_IR_BINDINGS_ROIALIGN_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

/**
 * Add bindings for the conv op.
 **/
void bindRoiAlign(py::module &m);

} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_ROIALIGN_HPP
