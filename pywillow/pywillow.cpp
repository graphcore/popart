#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <willow/l1.hpp>
#include <willow/loss.hpp>
#include <willow/nll.hpp>
#include <willow/willow.hpp>

namespace py = pybind11;
using namespace willow;

PYBIND11_MODULE(pywillow, m) {
  // py::module m("pywillow", "binding for willow");
  m.doc() = "binding for C++ willow library";

  py::class_<Loss> loss(m, "Loss");
  loss.def("input", &Loss::input);

  py::class_<NllLoss>(m, "NllLoss", loss)
      .def(py::init<TensorId, TensorId, TensorId>())
      .def("probsTensorId", &NllLoss::probsTensorId)
      .def("labelTensorId", &NllLoss::labelTensorId);

  py::class_<L1Loss>(m, "L1Loss", loss)
      .def(py::init<TensorId, TensorId, float>())
      .def("getInputId", &L1Loss::getInputId)
      .def("getLambda", &L1Loss::getLambda);

  py::class_<Willow>(m, "Willow")
      .def(py::init<std::string, const std::vector<Loss *> &>())
      .def("connect", &Willow::connect)
      .def("compile", &Willow::compile);
}
