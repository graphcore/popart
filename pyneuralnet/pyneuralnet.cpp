#include <neuralnet/neuralnet.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace neuralnet;

PYBIND11_MODULE(pyneuralnet, m) {
  // py::module m("pyneuralnet", "binding for neuralnet");
  m.doc() = "binding for C++ neuralnet library";
  py::class_<NeuralNet>(m, "NeuralNet")
      .def(py::init<std::string>())
      .def("connect", &NeuralNet::connect)
      .def("compile", &NeuralNet::compile);
}
