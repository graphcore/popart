// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "bindings/op/argminmax.hpp"
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
void bindArgMinMax(py::module &m) {
  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<ArgExtremaOp, Op, std::shared_ptr<ArgExtremaOp>>(sm,
                                                              "ArgExtremaOp")
      .def(py::init<const OperatorIdentifier &,
                    int64_t,
                    int64_t,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("axis"),
           py::arg("keepdims"),
           py::arg("settings"));

  py::class_<ArgMaxOp, ArgExtremaOp, std::shared_ptr<ArgMaxOp>>(sm, "ArgMaxOp")
      .def(py::init<const OperatorIdentifier &,
                    int64_t,
                    int64_t,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("axis"),
           py::arg("keepdims"),
           py::arg("settings"));

  py::class_<ArgMinOp, ArgExtremaOp, std::shared_ptr<ArgMinOp>>(sm, "ArgMinOp")
      .def(py::init<const OperatorIdentifier &,
                    int64_t,
                    int64_t,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("axis"),
           py::arg("keepdims"),
           py::arg("settings"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
