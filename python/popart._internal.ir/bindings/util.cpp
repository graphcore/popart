// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/util.hpp"

#include <initializer_list>
#include <pybind11/pybind11.h>
#include <popart/graph.hpp> // IWYU pragma: keep
#include <popart/util.hpp>  // IWYU pragma: keep

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindUtil(py::module &m) {
  m.def("addScope", &addScope);
  m.def("removeScope", &removeScope);
}

} // namespace ir
} // namespace _internal
} // namespace popart
