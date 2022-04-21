// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "bindings/op/pool.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindPool(py::module &m) {
  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<HasReceptiveFieldOp::ReceptiveOpAttributes>(
      sm, "ReceptiveOpAttributes")
      .def(py::init<>())
      .def_readwrite("pads", &HasReceptiveFieldOp::ReceptiveOpAttributes::pads)
      .def_readwrite("outPads",
                     &HasReceptiveFieldOp::ReceptiveOpAttributes::outPads)
      .def_readwrite("strides",
                     &HasReceptiveFieldOp::ReceptiveOpAttributes::strides)
      .def_readwrite("dilations",
                     &HasReceptiveFieldOp::ReceptiveOpAttributes::dilations)
      .def_readwrite("inDilations",
                     &HasReceptiveFieldOp::ReceptiveOpAttributes::inDilations)
      .def_readwrite("auto_pad",
                     &HasReceptiveFieldOp::ReceptiveOpAttributes::auto_pad)
      .def_readwrite("ceil_mode",
                     &HasReceptiveFieldOp::ReceptiveOpAttributes::ceil_mode);

  py::class_<HasReceptiveFieldOp,
             PyHasReceptiveOp<>,
             Op,
             std::shared_ptr<HasReceptiveFieldOp>>(sm, "HasReceptiveFieldOp")
      .def(py::init<const OperatorIdentifier &,
                    const HasReceptiveFieldOp::ReceptiveOpAttributes &,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("attributes"),
           py::arg("settings"));

  py::class_<AveragePoolOp,
             HasReceptiveFieldOp,
             std::shared_ptr<AveragePoolOp>>(sm, "AveragePoolOp")
      .def(py::init<const OperatorIdentifier &,
                    int64_t,
                    const std::vector<int64_t> &,
                    const HasReceptiveFieldOp::ReceptiveOpAttributes &,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("countIncludePad"),
           py::arg("kernelShape"),
           py::arg("attributes"),
           py::arg("settings"));

  py::class_<MaxPoolOp, HasReceptiveFieldOp, std::shared_ptr<MaxPoolOp>>(
      sm, "MaxPoolOp")
      .def(py::init<const OperatorIdentifier &,
                    const std::vector<int64_t> &,
                    int64_t,
                    const HasReceptiveFieldOp::ReceptiveOpAttributes &,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("kernelShape"),
           py::arg("storageOrder"),
           py::arg("attributes"),
           py::arg("settings"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
