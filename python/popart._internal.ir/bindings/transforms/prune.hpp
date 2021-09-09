// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_TRANSFORM_PRUNE_HPP
#define POPART__INTERNAL_IR_BINDINGS_TRANSFORM_PRUNE_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace transforms {

/**
 * Add bindings for `popart::Prune` class to pybind module.
 **/
void bindPrune(py::module &m);

} // namespace transforms
} // namespace ir
} // namespace _internal
} // namespace popart

#endif