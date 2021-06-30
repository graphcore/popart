// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/debugcontext.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <popart/debugcontext.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindDebugContext(py::module &m) {
  py::class_<SourceLocation>(m, "SourceLocation")
      .def(py::init<std::string, std::string, unsigned>(),
           py::arg("functionName"),
           py::arg("fileName"),
           py::arg("lineNumber"));

  py::class_<ProfileValue>(m, "ProfileValue")
      .def(py::init<ProfileValue::String>(), py::arg("init"));
  // Allow for string to be implicitly converted to a ProfileValue.
  py::implicitly_convertible<ProfileValue::String, ProfileValue>();

  // Unit test in tests/popart/debug_info_test.py.
  py::class_<DebugInfo>(m, "DebugInfo")
      .def(py::init<const DebugContext &, std::string>(),
           py::arg("debugContext"),
           py::arg("layer"))
      .def("setValue", &DebugInfo::setValue, py::arg("name"), py::arg("value"));

  py::class_<DebugContext>(m, "DebugContext")
      .def(py::init<std::string, SourceLocation>(),
           py::arg("name"),
           py::arg("loc"))
      .def(py::init([]() -> DebugContext {
        // This binding does the magic of getting the callee file & line number.
        auto inspect = pybind11::module::import("inspect");
        py::list s   = inspect.attr("stack")();

        auto callee         = s[0];
        py::str funcName    = callee.attr("function");
        py::str fileName    = callee.attr("filename");
        py::int_ lineNumber = callee.attr("lineno");

        return DebugContext(
            popart::SourceLocation(funcName, fileName, lineNumber));
      }))
      .def(py::init([](const std::string &name) -> DebugContext {
             auto inspect = pybind11::module::import("inspect");
             py::list s   = inspect.attr("stack")();

             auto callee         = s[0];
             py::str funcName    = callee.attr("function");
             py::str fileName    = callee.attr("filename");
             py::int_ lineNumber = callee.attr("lineno");

             return DebugContext(
                 name, popart::SourceLocation(funcName, fileName, lineNumber));
           }),
           py::arg("name"))
      .def(py::init([](const DebugInfo &di,
                       const std::string &name) -> DebugContext {
             auto inspect        = pybind11::module::import("inspect");
             py::list s          = inspect.attr("stack")();
             auto callee         = s[0];
             py::str funcName    = callee.attr("function");
             py::str fileName    = callee.attr("filename");
             py::int_ lineNumber = callee.attr("lineno");

             return DebugContext(
                 di,
                 name,
                 popart::SourceLocation(funcName, fileName, lineNumber));
           }),
           py::arg("debugInfo"),
           py::arg("name"));
  // Allow for string to be implicitly converted to a DebugContext.
  py::implicitly_convertible<std::string, DebugContext>();
}

} // namespace ir
} // namespace _internal
} // namespace popart
