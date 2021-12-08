// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/tensorinfo.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

namespace py = pybind11;
using namespace py::literals;

namespace popart {
namespace _internal {
namespace ir {

void bindTensorInfo(py::module &m) {
  py::enum_<DataType>(m, "DataType")
      .value("UINT8", DataType::UINT8)
      .value("INT8", DataType::INT8)
      .value("UINT16", DataType::UINT16)
      .value("INT16", DataType::INT16)
      .value("INT32", DataType::INT32)
      .value("INT64", DataType::INT64)
      .value("UINT32", DataType::UINT32)
      .value("UINT64", DataType::UINT64)
      .value("BOOL", DataType::BOOL)
      .value("FLOAT", DataType::FLOAT)
      .value("FLOAT16", DataType::FLOAT16)
      .value("BFLOAT16", DataType::BFLOAT16)
      .value("DOUBLE", DataType::DOUBLE)
      .value("COMPLEX64", DataType::COMPLEX64)
      .value("COMPLEX128", DataType::COMPLEX128)
      .value("STRING", DataType::STRING)
      .value("UNDEFINED", DataType::UNDEFINED);

  py::class_<TensorInfo>(m, "TensorInfo")
      .def(py::init<>())
      .def(py::init<DataType, const Shape &>(),
           py::arg("dataType"),
           py::arg("shape"))
      .def(py::init<DataType, const Shape &, const Shape &>(),
           py::arg("dataType"),
           py::arg("shape"),
           py::arg("metaShape"))
      .def(py::init<std::string, std::string>(),
           py::arg("dataType"),
           py::arg("shape"))
      .def(py::init<std::string, const Shape &>(),
           py::arg("dataType"),
           py::arg("shape"))
      .def("set",
           py::overload_cast<DataType>(&TensorInfo::set),
           py::arg("dataType"))
      .def("set",
           py::overload_cast<DataType, const Shape &>(&TensorInfo::set),
           py::arg("dataType"),
           py::arg("shape"))
      .def("set",
           py::overload_cast<DataType, const Shape &, const Shape &>(
               &TensorInfo::set),
           py::arg("dataType"),
           py::arg("shape"),
           py::arg("metaShape"))
      .def("shape", &TensorInfo::shape)
      .def("metaShape", &TensorInfo::metaShape)
      .def("rank", &TensorInfo::rank)
      .def("nelms", &TensorInfo::nelms)
      .def("nbytes", &TensorInfo::nbytes)
      .def("dim", &TensorInfo::dim, py::arg("i"))
      .def("dataType", &TensorInfo::dataType)
      .def("data_type", &TensorInfo::data_type)
      .def("data_type_lcase", &TensorInfo::data_type_lcase)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("set",
           py::overload_cast<DataType, const Shape &, const Shape &>(
               &TensorInfo::set));
}

} // namespace ir
} // namespace _internal
} // namespace popart
