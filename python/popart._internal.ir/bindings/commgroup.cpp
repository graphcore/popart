// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <initializer_list>
#include <pybind11/cast.h>
#include <pybind11/functional.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/commgroup.hpp>
#include <popart/docs/pydocs_popart_core.hpp>
#include <popart/replicagrouping.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindCommGroup(py::module &m) {

  py::enum_<CommGroupType>(m, "CommGroupType", DOC(popart, CommGroupType))
      .value("All", CommGroupType::All, DOC(popart, CommGroupType, All))
      .value("Consecutive",
             CommGroupType::Consecutive,
             DOC(popart, CommGroupType, Consecutive))
      .value("Orthogonal",
             CommGroupType::Orthogonal,
             DOC(popart, CommGroupType, Orthogonal))
      .value(
          "Ungrouped", CommGroupType::None, DOC(popart, CommGroupType, None));

  py::class_<CommGroup>(m, "CommGroup", DOC(popart, CommGroup))
      .def(py::init<>())
      .def(py::init<CommGroupType, unsigned>(),
           py::arg("type"),
           py::arg("replicaGroupSize"))
      .def(py::init<ReplicaGrouping>(), py::arg("grouping"))
      .def("toReplicaGrouping",
           &CommGroup::toReplicaGrouping,
           py::arg("numReplicas"))
      .def_readwrite("type", &CommGroup::type, DOC(popart, CommGroup, type))
      .def_readwrite("replicaGroupSize",
                     &CommGroup::replicaGroupSize,
                     DOC(popart, CommGroup, replicaGroupSize));
}

} // namespace ir
} // namespace _internal
} // namespace popart
