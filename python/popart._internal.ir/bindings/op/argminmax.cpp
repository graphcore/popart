// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "bindings/op/argminmax.hpp"

#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <stdint.h>

#include "popart/op.hpp"
#include "popart/op/argextrema.hpp"
#include "popart/op/argmax.hpp"
#include "popart/op/argmin.hpp"

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

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
