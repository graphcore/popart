// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/patterns/pattern.hpp"
#include "bindings/ir.hpp"

#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <popart/basicoptionals.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace patterns {

void bindPattern(py::module &m) {

  py::class_<Pattern, std::shared_ptr<Pattern>>(m, "Pattern")
      .def(py::init<>())
      .def("getPatternName", &Pattern::getPatternName);

  py::class_<PreAliasPattern,
             PyPreAliasPattern,
             Pattern,
             std::shared_ptr<PreAliasPattern>>(m, "PreAliasPattern")
      .def(py::init<>())
      .def("matches", &PreAliasPattern::matches)
      .def("apply", &PreAliasPattern::apply)
      .def("makeReplacementOpInIr", &PreAliasPattern::makeReplacementOpInIr)
      .def("touches",
           &PreAliasPattern::touches,
           py::return_value_policy::reference)
      .def("touchesAnchored", &PreAliasPattern::touchesAnchored);
}

} // namespace patterns
} // namespace ir
} // namespace _internal
} // namespace popart
