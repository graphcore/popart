// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "bindings/replicagrouping.hpp"

#include "popart/replicagrouping.hpp"
#include "pybind11/operators.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindReplicaGrouping(py::module &m) {

  py::class_<ReplicaGrouping>(m, "ReplicaGrouping")
      .def(py::init<unsigned, unsigned, unsigned>(),
           py::arg("numReplicas"),
           py::arg("stride"),
           py::arg("groupSize"))
      .def(py::init<unsigned>(), py::arg("numReplicas"))
      .def("getNumReplicas", &ReplicaGrouping::getNumReplicas)
      .def("getStride", &ReplicaGrouping::getStride)
      .def("getGroupSize", &ReplicaGrouping::getGroupSize)
      .def("getNumGroups", &ReplicaGrouping::getNumGroups)
      .def("getGroupAt", &ReplicaGrouping::getGroupAt, py::arg("replica"))
      .def("getIndexInGroupAt",
           &ReplicaGrouping::getIndexInGroupAt,
           py::arg("replica"))
      .def("getReplicaAt",
           &ReplicaGrouping::getReplicaAt,
           py::arg("group"),
           py::arg("index"))
      .def("getReplicasAt", &ReplicaGrouping::getReplicasAt, py::arg("group"))
      .def("getTranspose", &ReplicaGrouping::getTranspose)
      .def("str", &ReplicaGrouping::str)
      .def(py::self == py::self)
      .def(py::self != py::self);
}

} // namespace ir
} // namespace _internal
} // namespace popart
