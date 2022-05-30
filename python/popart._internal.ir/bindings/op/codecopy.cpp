// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "bindings/op/codecopy.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/op/exchange/codecopy.hpp>
#include <popart/tensorlocation.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensordebuginfo.hpp"

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {
namespace exchange {

// cppcheck-suppress constParameter // False positive for &m
void bindCodeCopy(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");
  sm = sm.def_submodule("exchange", "Python bindings for PopART exchange ops.");

  py::class_<RemoteCodeLoadOp, popart::Op, std::shared_ptr<RemoteCodeLoadOp>>(
      sm, "RemoteCodeLoadOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    const GraphId &,
                    const CodeMemoryType,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("graphid"),
           py::arg("destinationType"),
           py::arg("settings"));
}
} // namespace exchange
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
