#include <willow/willow.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace willow;

PYBIND11_MODULE(pywillow, m) {
  // py::module m("pywillow", "binding for willow");
  m.doc() = "binding for C++ willow library";
  py::class_<Willow>(m, "Willow")
      .def(py::init<std::string>())
      .def("connect", &Willow::connect)
      .def("compile", &Willow::compile);
}
