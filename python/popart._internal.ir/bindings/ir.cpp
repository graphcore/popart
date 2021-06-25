// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/ir.hpp"

#include <pybind11/pybind11.h>
#include <popart/ir.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindIr(py::module_ &m) { py::class_<Ir>(m, "Ir").def(py::init<>()); }

} // namespace ir
} // namespace _internal
} // namespace popart
