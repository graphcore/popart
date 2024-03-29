// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OP_MATMUL_HPP_
#define POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OP_MATMUL_HPP_

#include <pybind11/pybind11.h>
#include <popart/names.hpp>
#include <popart/op/matmul.hpp>

#include "bindings/op.hpp"

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

/**
 * This is a trampoline class; discussed extensively here:
 * https://pybind11.readthedocs.io/en/stable/advanced/classes.html#classes
 *  As the base Op class has virtual and pure virtual methods, we must create
 * this in-between class that redirects virtual calls back to Python.
 *
 * We create another template on top of the PyOp template to account for the
 * additional pure virtual methods getExpanded[Lhs|Rhs]Shape().
 *
 * \tparam BaseOp The op type. This is a template in case we have other Op
 * classes that require the trampoline. Defaults to MatMulBaseOp.
 */
template <class BaseOp = MatMulBaseOp> class PyMatMulOp : public PyOp<BaseOp> {
public:
  using PyOp<BaseOp>::PyOp;

  Shape getExpandedLhsShape() const override {
    PYBIND11_OVERRIDE_PURE(
        Shape,  /* Return type */
        BaseOp, /* Parent class */
        // cppcheck-suppress syntaxError // Variadic macro requires >=1 argument
        getExpandedLhsShape, /* Name of function in C++ (must
                                match Python name) */
    );
  }
  Shape getExpandedRhsShape() const override {
    PYBIND11_OVERRIDE_PURE(Shape,               /* Return type */
                           BaseOp,              /* Parent class */
                           getExpandedRhsShape, /* Name of function in C++ (must
                                                   match Python name) */
    );
  }
};

/**
 * Add bindings for the matmul op.
 **/
void bindMatmul(py::module &m);

} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_OP_MATMUL_HPP_
