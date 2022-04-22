// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_PATTERN_HPP
#define POPART__INTERNAL_IR_BINDINGS_PATTERN_HPP

#include <pybind11/pybind11.h>
#include <vector>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp> // IWYU pragma: keep

namespace py = pybind11;

namespace popart {
class Op;

namespace _internal {
namespace ir {
namespace patterns {

/**
 * This is a trampoline class; discussed extensively here:
 * https://pybind11.readthedocs.io/en/stable/advanced/classes.html#classes
 *  As the base Op class has virtual and pure virtual methods, we must create
 * this in-between class that redirects virtual calls back to Python.
 *
 **/
class PyPreAliasPattern : public PreAliasPattern {
public:
  using PreAliasPattern ::PreAliasPattern;

  bool matches(Op *op) const override {
    PYBIND11_OVERLOAD_PURE(
        bool,            /* Return type */
        PreAliasPattern, /* Parent class */
        matches,         /* Name of function in C++ (must match Python name) */
        op);
  }
  bool apply(Op *op) const override {
    PYBIND11_OVERLOAD_PURE(
        bool,            /* Return type */
        PreAliasPattern, /* Parent class */
        apply,           /* Name of function in C++ (must match Python name) */
        op);
  }
  std::vector<const Tensor *> touches(Op *op) const override {
    PYBIND11_OVERLOAD_PURE(
        std::vector<const Tensor *>, /* Return type */
        PreAliasPattern,             /* Parent class */
        touches, /* Name of function in C++ (must match Python name) */
        op);
  }
};

template <class BasePattern>
class PyDerivedPreAliasPattern : public BasePattern {
public:
  using BasePattern::BasePattern;

  bool matches(Op *op) const override {
    PYBIND11_OVERRIDE(
        bool,        /* Return type */
        BasePattern, /* Parent class */
        matches,     /* Name of function in C++ (must match Python name) */
        op);
  }
  bool apply(Op *op) const override {
    PYBIND11_OVERRIDE(
        bool,        /* Return type */
        BasePattern, /* Parent class */
        apply,       /* Name of function in C++ (must match Python name) */
        op);
  }
  std::vector<const Tensor *> touches(Op *op) const override {
    PYBIND11_OVERRIDE(
        std::vector<const Tensor *>, /* Return type */
        BasePattern,                 /* Parent class */
        touches, /* Name of function in C++ (must match Python name) */
        op);
  }
};

/**
 * Add bindings for `popart::Op` class to pybind module.
 **/
void bindPattern(py::module &m);

} // namespace patterns
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_PATTERN_HPP
