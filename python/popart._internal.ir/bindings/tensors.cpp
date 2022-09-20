// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/tensors.hpp"

#include <initializer_list>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <string>
#include <vector>
#include <popart/tensors.hpp>

#include "../../popart/shared_cpp/np_utils.hpp"
#include "popart/debugcontext.hpp"
#include "popart/graph.hpp" // IWYU pragma: keep
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"

namespace py = pybind11;

namespace popart {
class InputSettings;
class Scope;
class TensorInfo;

namespace _internal {
namespace ir {

void bindTensors(py::module_ &m) {
  py::class_<Tensors>(m, "Tensors")
      .def(py::init<Graph &>())
      .def("get", &Tensors::get, py::return_value_policy::reference)
      .def("remove", &Tensors::remove)
      .def("contains",
           py::overload_cast<TensorId>(&Tensors::contains, py::const_))
      .def("contains",
           py::overload_cast<TensorId, const Scope &>(&Tensors::contains,
                                                      py::const_))
      .def("n", &Tensors::n)
      .def("find", &Tensors::find)
      .def(
          "addVarInit",
          [](Tensors &self,
             const TensorId &tid,
             const TensorInfo &tinfo,
             py::array data,
             const DebugContext &dc) {
            data = makeContiguous(data);
            self.addVarInit(tid, tinfo, data.request().ptr, dc);
          },
          py::arg("tensorId"),
          py::arg("tensorInfo"),
          py::arg("data"),
          py::arg("debugContext") = std::string())
      .def(
          "addConstInit",
          [](Tensors &self,
             const TensorId &tid,
             const TensorInfo &tinfo,
             py::array data,
             const DebugContext &dc) {
            data = makeContiguous(data);
            self.addConstInit(tid, tinfo, data.request().ptr, dc);
          },
          py::arg("tensorId"),
          py::arg("tensorInfo"),
          py::arg("data"),
          py::arg("debugContext") = std::string())
      .def("makeConstInit",
           [](Tensors &self, const TensorId &tid, py::array data) {
             data = makeContiguous(data);
             self.makeConstInit(tid, data.request().ptr);
           })
      .def(
          "addStream",
          py::overload_cast<TensorId, const TensorInfo &, const DebugContext &>(
              &Tensors::addStream),
          py::arg("tensorId"),
          py::arg("tensorInfo"),
          py::arg("debugContext") = std::string())
      .def("addStream",
           py::overload_cast<TensorId,
                             const TensorInfo &,
                             const InputSettings &,
                             const DebugContext &>(&Tensors::addStream),
           py::arg("tensorId"),
           py::arg("tensorInfo"),
           py::arg("inputSettings"),
           py::arg("debugContext") = std::string())
      .def("addActGrad",
           &Tensors::addActGrad,
           py::arg("tensorId"),
           py::arg("dc") = std::string())
      .def("getIds", &Tensors::getIds)
      .def("getOfType",
           py::overload_cast<TensorType>(&Tensors::getOfType, py::const_),
           py::return_value_policy::reference)
      .def("getOfType",
           py::overload_cast<const std::vector<TensorType> &>(
               &Tensors::getOfType, py::const_),
           py::return_value_policy::reference)
      .def("getAllTensorIds", &Tensors::getAllTensorIds)
      .def("getNoProducerIds", &Tensors::getNoProducerIds)
      .def("append", &Tensors::append)
      .def("removeIsolated",
           &Tensors::removeIsolated,
           py::arg("retainUsedIOTensors") = false,
           py::arg("retainAllIOTensors")  = false,
           py::arg("retainVarTensors")    = false,
           py::arg("retainConstTensors")  = false);
}

} // namespace ir
} // namespace _internal
} // namespace popart
