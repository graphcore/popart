// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/remotebufferinfo.hpp"

#include <pybind11/pybind11.h>

#include <popart/ir.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindRemoteBufferInfo(py::module &m) {
  py::class_<RemoteBufferInfo>(m, "RemoteBufferInfo")
      .def(py::init<TensorInfo &, uint64_t &>(),
           py::arg("info"),
           py::arg("repeats"))
      .def_readonly("TensorInfo", &RemoteBufferInfo::info)
      .def_readonly("repeats", &RemoteBufferInfo::repeats);
}

} // namespace ir
} // namespace _internal
} // namespace popart
