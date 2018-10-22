#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <willow/l1.hpp>
#include <willow/loss.hpp>
#include <willow/nll.hpp>
#include <willow/optimizer.hpp>
#include <willow/willownet.hpp>

// note to developers: be very careful
// about exposing functions which return pointers
// or references, python ref counter might behave
// unexpectedly. More info:
// https://pybind11.readthedocs.io/en/stable/advanced/functions.html

namespace py = pybind11;
using namespace willow;

PYBIND11_MODULE(pywillow, m) {
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

  py::class_<BaseSGD> basesgd(m, "BaseSGD", optimizer);
  // Note that we do not define a constructor, as it is a virtual class
  basesgd.def("learnRate", &BaseSGD::learnRate);

  // The Optimizer classes which are non-virtual:
  py::class_<SGD>(m, "SGD", basesgd).def(py::init<float>());
  py::class_<ConstSGD>(m, "ConstSGD", basesgd).def(py::init<float>());

  py::class_<WillowNet>(m, "WillowNet")
      .def(py::init<std::string,
                    const EarlyInfo &,
                    const DataFlow &,
                    const std::vector<Loss *> &,
                    const Optimizer *,
                    const std::vector<TensorId> &,
                    std::string,
                    const std::vector<std::string> &>())
      .def("updateOptimizer", &WillowNet::updateOptimizer)
      .def("setDevice", &WillowNet::setDevice)
      .def("prepareDevice", &WillowNet::prepareDevice)
      .def("weightsFromHost", &WillowNet::weightsFromHost)
      .def("optimizerFromHost", &WillowNet::optimizerFromHost)
      .def("step", &WillowNet::step)
      .def("modelToHost", &WillowNet::modelToHost);

  // This does not seem to work :/
  // Thoroughly read
  // https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
  // and looked through pybind11 code, still can't get it to bight
  auto ex5 = py::register_exception<willow::error>(m, "WillowException");
}
