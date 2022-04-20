// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/ipucopy.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/op/ipucopy.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensordebuginfo.hpp"

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
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
           py::overload_cast<InIndex, TensorId, VGraphId>(
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
