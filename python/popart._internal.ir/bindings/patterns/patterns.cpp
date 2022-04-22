// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/patterns/patterns.hpp"

#include <initializer_list>
#include <memory>
#include <pybind11/attr.h>
#include <pybind11/functional.h> // IWYU pragma: keep
#include <pybind11/operators.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <string>
#include <vector>
#include <popart/patterns/patterns.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace patterns {

void bindPatterns(py::module &m) {

  py::enum_<PatternsLevel>(m, "PatternsLevel", py::module_local())
      .value("NoPatterns", PatternsLevel::NoPatterns)
      .value("Minimal", PatternsLevel::Minimal)
      .value("Default", PatternsLevel::Default)
      .value("All", PatternsLevel::All);

  py::class_<Patterns, std::shared_ptr<Patterns>>(
      m, "Patterns", py::module_local())
      .def(py::init<>())
      .def(py::init<PatternsLevel>())
      .def(py::init<std::vector<std::string>>())
      .def("enableRuntimeAsserts",
           &Patterns::enableRuntimeAsserts,
           py::return_value_policy::reference)
      .def("isPatternEnabled",
           static_cast<bool (Patterns::*)(const std::string &)>(
               &Patterns::isPatternEnabled))
      .def("enablePattern",
           static_cast<Patterns &(Patterns::*)(const std::string &, bool)>(
               &Patterns::enablePattern))
      .def_static("getAllPreAliasPatternNames",
                  &Patterns::getAllPreAliasPatternNames)
      .def_static("isMandatory",
                  py::overload_cast<std::string &>(&Patterns::isMandatory));
}

} // namespace patterns
} // namespace ir
} // namespace _internal
} // namespace popart
