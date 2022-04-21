// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iterator>
#include <map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../popart/shared_cpp/np_utils.hpp"
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/topocons.hpp>

namespace py = pybind11;

namespace popart {
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
