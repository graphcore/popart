// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <pybind11/stl.h> // IWYU pragma: keep

#include "bindings/patterns/pattern.hpp"

#include <initializer_list>
#include <memory>
#include <vector>

#include <pybind11/functional.h> // IWYU pragma: keep
#include <pybind11/operators.h>  // IWYU pragma: keep

#include <pybind11/pybind11.h>
#include <popart/op.hpp> // IWYU pragma: keep
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp> // IWYU pragma: keep

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
