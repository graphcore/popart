// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <initializer_list>
#include <pybind11/numpy.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/topocons.hpp>

namespace py = pybind11;

namespace popart {
class Op;

namespace _internal {
namespace ir {

void bindTopoCons(py::module &m) {
  py::class_<TopoCons> tc(m, "TopoCons");

  tc.def("insert", py::overload_cast<Op *, Op *, bool>(&TopoCons::insert))
      .def_static("transferToSubgraph", &TopoCons::transferToSubgraph);
}

} // namespace ir
} // namespace _internal
} // namespace popart
