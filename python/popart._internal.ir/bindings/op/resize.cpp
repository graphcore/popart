// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <string>
#include <vector>
#include <popart/op/resize.hpp>

#include "bindings/op/resize.hpp"
#include "popart/op.hpp"

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;
namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindResize(py::module &m) {
  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::enum_<ResizeMode>(sm, "ResizeMode")
      .value("Nearest", ResizeMode::Nearest)
      .value("Linear", ResizeMode::Linear)
      .value("Cubic", ResizeMode::Cubic)
      .value("N", ResizeMode::N);

  py::enum_<ResizeNearestMode>(sm, "ResizeNearestMode")
      .value("RoundPreferFloor", ResizeNearestMode::RoundPreferFloor)
      .value("RoundPreferCeil", ResizeNearestMode::RoundPreferCeil)
      .value("Floor", ResizeNearestMode::Floor)
      .value("Ceil", ResizeNearestMode::Ceil)
      .value("Pytorch", ResizeNearestMode::Pytorch)
      .value("N", ResizeNearestMode::N);

  py::enum_<ResizeCoordinateTransformationMode>(
      sm, "ResizeCoordinateTransformationMode")
      .value("HalfPixel", ResizeCoordinateTransformationMode::HalfPixel)
      .value("PytorchHalfPixel",
             ResizeCoordinateTransformationMode::PytorchHalfPixel)
      .value("AlignCorners", ResizeCoordinateTransformationMode::AlignCorners)
      .value("Asymmetric", ResizeCoordinateTransformationMode::Asymmetric)
      .value("TfCropAndResize",
             ResizeCoordinateTransformationMode::TfCropAndResize)
      .value("N", ResizeCoordinateTransformationMode::N);

  py::class_<ResizeOp, Op, std::shared_ptr<ResizeOp>>(sm, "ResizeOp")
      .def(py::init<const OperatorIdentifier &,
                    const Op::Settings &,
                    ResizeMode &,
                    const std::vector<float> &>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("mode"),
           py::arg("scales"))
      .def(py::init<const OperatorIdentifier &,
                    const Op::Settings &,
                    ResizeMode &,
                    const std::vector<float> &,
                    ResizeNearestMode &,
                    ResizeCoordinateTransformationMode &>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("mode"),
           py::arg("scales"),
           py::arg("nearestMode"),
           py::arg("coordinateTransformationMode"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
