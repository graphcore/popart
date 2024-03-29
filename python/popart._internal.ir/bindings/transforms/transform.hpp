// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_TRANSFORMS_TRANSFORM_HPP_
#define POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_TRANSFORMS_TRANSFORM_HPP_

#include <cstddef>
#include <pybind11/pybind11.h>
#include <string>
#include <popart/graph.hpp> // IWYU pragma: keep
#include <popart/transforms/transform.hpp>

namespace py = pybind11;

namespace popart {

namespace _internal {
namespace ir {
namespace transforms {
/**
 * This is a trampoline class; discussed extensively here:
 * https://pybind11.readthedocs.io/en/stable/advanced/classes.html#classes
 *  As the base Op class has virtual and pure virtual methods, we must create
 * this in-between class that redirects virtual calls back to Python.
 *
 **/

template <class TransformBase = Transform>
class PyTransform : public TransformBase {
public:
  using TransformBase::TransformBase;

  bool apply(Graph &graph) const override {
    PYBIND11_OVERRIDE_PURE(bool,          /* Return type */
                           TransformBase, /* Parent class */
                           apply,         /* Name of function */
                           graph);
  }
  std::size_t getId() const override {
    PYBIND11_OVERRIDE_PURE(
        std::size_t,   /* Return type */
        TransformBase, /* Parent class */
        // cppcheck-suppress syntaxError // Variadic macro requires >=1 argument
        getId, /* Name of function */
    );
  }
  std::string getName() const override {
    PYBIND11_OVERRIDE_PURE(std::string,   /* Return type */
                           TransformBase, /* Parent class */
                           getName,       /* Name of function */
    );
  }
};

/**
 * Add bindings for `popart::Transform` class to pybind module.
 **/
void bindTransform(py::module &m);

} // namespace transforms
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_TRANSFORMS_TRANSFORM_HPP_
