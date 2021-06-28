// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/scope.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/scope.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindScope(py::module &m) {
  py::class_<Scope>(m, "Scope")
      .def(py::init<>())
      .def_static("delimiter", &Scope::delimiter)
      .def("empty", &Scope::empty)
      .def("pop", &Scope::pop)
      // NOTE: This binding won't compile without wrapping it in a lambda.
      .def("getCommonParent",
           [](Scope &self, const Scope &otherScope) {
             return self.getCommonParent(otherScope);
           })
      .def("depth", &Scope::depth)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("str", &Scope::str)
      .def(py::self / std::string())
      .def("isSubscope", &Scope::isSubscope)
      // NOTE: Unable to overload as "getCommonParent" because it would overload
      // an instance method with a static one. Also, this binding won't compile
      // without wrapping it in a lambda.
      .def_static("getCommonParent_static", [](const std::vector<Op *> &ops) {
        return Scope::getCommonParent(ops);
      });
  ;
}

} // namespace ir
} // namespace _internal
} // namespace popart
