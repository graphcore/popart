// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/concat.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include "popart/op.hpp"
#include "popart/op/concat.hpp"
#include "popart/region.hpp" // IWYU pragma: keep

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindConcat(py::module &m) {
  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<ConcatOp, popart::Op, std::shared_ptr<ConcatOp>>(sm, "ConcatOp")
      .def(
          py::init<const OperatorIdentifier &, int64_t, const Op::Settings &>(),
          py::arg("opid"),
          py::arg("axis"),
          py::arg("settings"));

  py::class_<ConcatInplaceOp, popart::Op, std::shared_ptr<ConcatInplaceOp>>(
      sm, "ConcatInplaceOp")
      .def(py::init<int64_t, const Op::Settings &>(),
           py::arg("axis"),
           py::arg("settings"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
