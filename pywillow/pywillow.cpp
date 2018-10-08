#include <willow/willow.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace willow;

py::array_t<double> floater() {
  py::array_t<double> x(100);
  auto b = x.request();
  auto d = static_cast<double *>(b.ptr);
  for (int i = 0; i < 100; ++i) {
    d[i] = i*i;
  }
  return x;
}

void doubler(py::array_t<double> & x){
  auto b = x.request();
  auto d = static_cast<double *>(b.ptr);
  for (int i = 0; i < b.size; ++i) {
    d[i] = 2*d[i];
  }
}

PYBIND11_MODULE(pywillow, m) {
  // py::module m("pywillow", "binding for willow");
  m.doc() = "binding for C++ willow library";
  py::class_<Willow>(m, "Willow")
      .def(py::init<std::string>())
      .def("connect", &Willow::connect)
      .def("compile", &Willow::compile);

  m.def("floater", &floater, "A function which returns hope");
  m.def("double", &doubler, "A function which returns hope");


}
