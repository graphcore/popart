// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/tensors.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/graph.hpp>
#include <popart/tensors.hpp>

#include "../../popart/shared_cpp/np_utils.hpp"

namespace py = pybind11;

namespace popart {
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
      // TODO add test, see T42141, requires VectorAndSet bindings
      // .def("getConstIds",
      //      &Tensors::getConstIds,
      //      py::return_value_policy::reference)
      // TODO add test, see T42141, requires VectorAndSet bindings for testing
      // .def("insertConstId", &Tensors::insertConstId)
      .def("removeIsolated",
           &Tensors::removeIsolated,
           py::arg("retainIoTensors")    = false,
           py::arg("retainVarTensors")   = false,
           py::arg("retainConstTensors") = false)
      // Not sure we can do this from the python api
      // TensorId moveIntoTensors(std::unique_ptr<Tensor> tensor);
      ;
}

} // namespace ir
} // namespace _internal
} // namespace popart
