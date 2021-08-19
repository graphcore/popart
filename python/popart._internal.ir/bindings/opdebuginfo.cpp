// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/opdebuginfo.hpp"
#include "bindings/op.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/opdebuginfo.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindOpDebugInfo(py::module &m) {

  py::class_<OpDebugInfo>(m, "OpDebugInfo", py::module_local())
      .def(py::init<const DebugContext &, const Op &>())
      .def("finalize", &OpDebugInfo::finalize);
}

} // namespace ir
} // namespace _internal
} // namespace popart
