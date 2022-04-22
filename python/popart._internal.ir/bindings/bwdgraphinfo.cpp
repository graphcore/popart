// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/bwdgraphinfo.hpp"

#include <initializer_list>
#include <pybind11/cast.h>       // IWYU pragma: keep
#include <pybind11/functional.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <vector>
#include <popart/bwdgraphinfo.hpp>

#include "popart/graphid.hpp"
#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindBwdGraphInfo(py::module &m) {

  py::enum_<ExpectedConnectionType>(m, "ExpectedConnectionType")
      .value("Fwd", ExpectedConnectionType::Fwd)
      .value("FwdGrad", ExpectedConnectionType::FwdGrad);

  py::class_<ExpectedConnection>(m, "ExpectedConnection")
      .def(py::init<TensorId, ExpectedConnectionType>())
      .def_readwrite("fwdId", &ExpectedConnection::fwdId)
      .def_readwrite("type", &ExpectedConnection::type);

  py::class_<BwdGraphInfo>(m, "BwdGraphInfo")
      .def(py::init<GraphId,
                    std::vector<ExpectedConnection>,
                    std::vector<ExpectedConnection>>())
      .def_readwrite("bwdGraphId", &BwdGraphInfo::bwdGraphId)
      .def_readwrite("expectedInputs", &BwdGraphInfo::expectedInputs)
      .def_readwrite("expectedOutputs", &BwdGraphInfo::expectedOutputs);
}

} // namespace ir
} // namespace _internal
} // namespace popart
