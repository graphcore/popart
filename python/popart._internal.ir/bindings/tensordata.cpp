// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "bindings/tensordata.hpp"
#include "../../popart/shared_cpp/np_utils.hpp"
#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <popart/tensordata.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindTensorData(py::module &m) {
  py::class_<TensorData>(m, "TensorData")
      .def(py::init([](const TensorInfo &info, py::array data) {
             data = makeContiguous(data);
             return TensorData(info, data.request().ptr);
           }),
           py::arg("tensorInfo"),
           py::arg("src"))
      .def(
          "resetData",
          [](TensorData &self, const TensorInfo &info, py::array data) {
            data = makeContiguous(data);
            self.resetData(info, data.request().ptr);
          },
          py::arg("tensorInfo"),
          py::arg("src"));
}

} // namespace ir
} // namespace _internal
} // namespace popart
