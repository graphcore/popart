// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "bindings/op/conv.hpp"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <string>
#include <vector>

#include "popart/attributes.hpp"
#include "popart/op.hpp"
#include "popart/op/conv.hpp"
#include "popart/op/convbase.hpp"
#include "popart/op/receptive.hpp"
#include "popart/vendored/optional.hpp"

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindConv(py::module &m) {
  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::enum_<AutoPad>(sm, "AutoPad")
      .value("NOTSET", AutoPad::NOTSET)
      .value("SAME_UPPER", AutoPad::SAME_UPPER)
      .value("SAME_LOWER", AutoPad::SAME_LOWER)
      .value("VALID", AutoPad::VALID);

  py::class_<Attributes>(sm, "Attributes").def(py::init<>());

  py::class_<MultiConvOptions>(sm, "MultiConvOptions")
      .def(py::init<const std::map<std::string, std::string>,
                    const Attributes &>(),
           py::arg("sessionConvOptions"),
           py::arg("attr"))
      .def_readwrite("availableMemoryProportions",
                     &MultiConvOptions::availableMemoryProportions)
      .def_readwrite("partialsTypes", &MultiConvOptions::partialsTypes)
      .def_readwrite("planType", &MultiConvOptions::planType)
      .def_readwrite("perConvReservedTiles",
                     &MultiConvOptions::perConvReservedTiles)
      .def_readwrite("cycleBackOff", &MultiConvOptions::cycleBackOff)
      .def_readwrite("enableConvDithering",
                     &MultiConvOptions::enableConvDithering);

  py::class_<MultiConvBaseOp, Op, std::shared_ptr<MultiConvBaseOp>>(
      sm, "MultiConvBaseOp")
      .def(py::init<const OperatorIdentifier &,
                    const Op::Settings &,
                    std::vector<int64_t>,
                    std::vector<int64_t>,
                    std::vector<int64_t>,
                    const AutoPad &,
                    const MultiConvOptions &>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("flatStrides"),
           py::arg("flatPads"),
           py::arg("flatDilations"),
           py::arg("padType"),
           py::arg("convOpts"));

  py::class_<ConvOp, MultiConvBaseOp, std::shared_ptr<ConvOp>>(sm, "ConvOp")
      .def(py::init<const OperatorIdentifier &,
                    const Op::Settings &,
                    std::vector<int64_t>,
                    std::vector<int64_t>,
                    std::vector<int64_t>,
                    int64_t,
                    const AutoPad &,
                    const MultiConvOptions &>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("strides"),
           py::arg("pads"),
           py::arg("dilations"),
           py::arg("group"),
           py::arg("padType"),
           py::arg("convOpts"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
