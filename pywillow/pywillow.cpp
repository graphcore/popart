#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <willow/graph.hpp>
#include <willow/l1.hpp>
#include <willow/loss.hpp>
#include <willow/nll.hpp>

// note to developers: be very careful
// about exposing functions which return pointers
// or references, python ref counter might behave
// unexpectedly. More info:
// https://pybind11.readthedocs.io/en/stable/advanced/functions.html

namespace py = pybind11;
using namespace willow;

PYBIND11_MODULE(pywillow, m) {
  // py::module m("pywillow", "binding for willow");
  m.doc() = "binding for C++ willow library";

  py::class_<DataFlow>(m, "DataFlow")
      .def(py::init<int, int, const std::vector<TensorId> &>(),
           py::arg("Batches processed between returning anchors"),
           py::arg("Batch size"),
           py::arg("Anchor tensors (tensors to return)"))
      .def("nAnchors", &DataFlow::nAnchors);

  py::class_<TensorInfo>(m, "TensorInfo")
      .def(py::init<std::string, const std::vector<int64_t> &>());

  py::class_<EarlyInfo>(m, "EarlyInfo")
      .def(py::init<>())
      .def("addInfo", &EarlyInfo::addInfo)
      .def("getInfo", &EarlyInfo::hasInfo)
      .def("hasInfo", &EarlyInfo::hasInfo);

  py::class_<Loss> loss(m, "Loss");
  loss.def("input", &Loss::input);

  py::class_<NllLoss>(m, "NllLoss", loss)
      .def(py::init<TensorId, TensorId, TensorId>())
      .def("probsTensorId", &NllLoss::probsTensorId)
      .def("labelTensorId", &NllLoss::labelTensorId);

  // TODO : document all the functions like this one
  py::class_<L1Loss>(m, "L1Loss", loss)
      .def(py::init<TensorId, TensorId, float>(),
           py::arg("The ID of the input tensor"),
           py::arg("The ID of the output tensor"),
           py::arg("lambda"))
      .def("getInputId", &L1Loss::getInputId)
      .def("getLambda", &L1Loss::getLambda);

  py::class_<Optimizer> optimizer(m, "Optimizer");
  // optimizer.def(py::init<>());

  py::class_<SGD>(m, "SGD", optimizer)
      .def(py::init<float>())
      .def("learnRate", &SGD::learnRate);

  py::class_<Graph>(m, "Graph")
      .def(py::init<std::string,
                    const EarlyInfo &,
                    const DataFlow &,
                    const std::vector<Loss *> &,
                    const Optimizer *,
                    const std::vector<TensorId> &,
                    std::string>());
}
