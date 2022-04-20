// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include "bindings/op/matmul.hpp"
#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include <popart/op/matmul.hpp>

namespace nonstd {
namespace optional_lite {
template <typename T> class optional;
} // namespace optional_lite
} // namespace nonstd

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindMatmul(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::enum_<popart::MatMulPartialsType>(sm, "MatMulPartialsType")
      .value("HALF", MatMulPartialsType::HALF)
      .value("FLOAT", MatMulPartialsType::FLOAT);

  py::enum_<MatMulBaseOp::SerialiseSettings::Mode>(
      sm, "SerialiseSettingsMode", py::module_local())
      .value("NoSerialisation", MatMulBaseOp::SerialiseSettings::Mode::None)
      .value("InputChannels",
             MatMulBaseOp::SerialiseSettings::Mode::InputChannels)
      .value("ReducingDim", MatMulBaseOp::SerialiseSettings::Mode::ReducingDim)
      .value("OutputChannels",
             MatMulBaseOp::SerialiseSettings::Mode::OutputChannels);

  py::class_<MatMulBaseOp::SerialiseSettings>(sm, "SerialiseSettings")
      .def(py::init<>())
      .def_readwrite("mode", &MatMulBaseOp::SerialiseSettings::mode)
      .def_readwrite("factor", &MatMulBaseOp::SerialiseSettings::factor)
      .def_readwrite("keep_precision",
                     &MatMulBaseOp::SerialiseSettings::keep_precision);

  py::class_<MatMulBaseOp, PyMatMulOp<>, Op, std::shared_ptr<MatMulBaseOp>>(
      sm, "MatMulBaseOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    const Op::Settings &,
                    const popart::MatMulBaseOp::Phase &,
                    const nonstd::optional<float> &,
                    const popart::MatMulBaseOp::SerialiseSettings &,
                    const popart::OptionalDataType &,
                    const popart::MatMulPartialsType &,
                    const bool &>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("phase"),
           py::arg("availableMemoryProportion"),
           py::arg("serialization"),
           py::arg("outputType"),
           py::arg("partialsType"),
           py::arg("enableFullyConnectedPass"))
      .def("setAvailableMemoryProportion",
           &MatMulBaseOp::setAvailableMemoryProportion);

  py::class_<MatMulOp, MatMulBaseOp, std::shared_ptr<MatMulOp>>(sm, "MatMulOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    const Op::Settings &,
                    const nonstd::optional<float> &,
                    const popart::MatMulBaseOp::SerialiseSettings &,
                    const popart::OptionalDataType &,
                    const popart::MatMulPartialsType &>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("availableMemoryProportion"),
           py::arg("serialization"),
           py::arg("outputType"),
           py::arg("partialsType"));

  py::class_<MatMulBaseGradOp,
             MatMulBaseOp,
             PyMatMulOp<MatMulBaseGradOp>,
             std::shared_ptr<MatMulBaseGradOp>>(sm, "MatMulBaseGradOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    const popart::MatMulOp &,
                    popart::MatMulBaseOp::Phase>(),
           py::arg("opid"),
           py::arg("fwdOp"),
           py::arg("phase"));
  py::class_<MatMulLhsGradOp,
             MatMulBaseGradOp,
             PyMatMulOp<MatMulLhsGradOp>,
             std::shared_ptr<MatMulLhsGradOp>>(sm, "MatMulLhsGradOp")
      .def(py::init<const popart::MatMulOp &>(), py::arg("op"));
  // Do not bind copy constructors

  py::class_<MatMulRhsGradOp,
             MatMulBaseGradOp,
             PyMatMulOp<MatMulRhsGradOp>,
             std::shared_ptr<MatMulRhsGradOp>>(sm, "MatMulRhsGradOp")
      .def(py::init<const popart::MatMulOp &>(), py::arg("op"));
  // Do not bind copy constructors
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
