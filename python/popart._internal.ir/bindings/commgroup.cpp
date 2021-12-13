// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/bwdgraphinfo.hpp"
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/commgroup.hpp>

#include <popart/docs/pydocs_popart_core.hpp>

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
      .def_readwrite("type", &CommGroup::type, DOC(popart, CommGroup, type))
      .def_readwrite("replicaGroupSize",
                     &CommGroup::replicaGroupSize,
                     DOC(popart, CommGroup, replicaGroupSize));
}

} // namespace ir
} // namespace _internal
} // namespace popart
