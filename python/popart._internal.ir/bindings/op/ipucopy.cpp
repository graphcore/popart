// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/ipucopy.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/op/ipucopy.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

void bindIpuCopy(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<IpuCopyOp, popart::Op, std::shared_ptr<IpuCopyOp>>(sm, "IpuCopyOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    uint64_t,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("destIpu"),
           py::arg("settings"))
      .def("connectInTensor",
           py::overload_cast<InIndex, TensorId, uint64_t>(
               &IpuCopyOp::connectInTensor))
      .def("getDestIpu", &IpuCopyOp::getDestIpu)
      .def("getSourceIpu",
           py::overload_cast<>(&IpuCopyOp::getSourceIpu, py::const_))
      .def("getSourceIpu",
           py::overload_cast<const TensorId &>(&IpuCopyOp::getSourceIpu,
                                               py::const_))
      .def("getMinSourceIpu", &IpuCopyOp::getMinSourceIpu)
      .def("getMaxSourceIpu", &IpuCopyOp::getMaxSourceIpu);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
