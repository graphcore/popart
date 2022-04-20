// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_POOL_HPP
#define POPART__INTERNAL_IR_BINDINGS_POOL_HPP

#include "bindings/op.hpp"
#include "popart/op/receptive.hpp"
#include <pybind11/pybind11.h>
#include <popart/names.hpp>

#include <cstdint>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

/**
 * As the base Op class has virtual and pure virtual methods, we must create
 * this in-between class that redirects virtual calls back to Python.
 *
 * We create another template on top of the PyOp template to account for the
 * additional pure virtual methods getSpatialK, getNOutChans and setup0.
 */
template <class FieldOp = HasReceptiveFieldOp>
class PyHasReceptiveOp : public PyOp<FieldOp> {

public:
  using PyOp<FieldOp>::PyOp;
  Shape getSpatialK() const override {

    PYBIND11_OVERRIDE_PURE(
        Shape,   /* Return type */
        FieldOp, /* Parent class */
        // cppcheck-suppress syntaxError
        getSpatialK, /* Name of function in C++ (must match Python name) */
    );
  }

  int64_t getNOutChans() const override {
    PYBIND11_OVERRIDE_PURE(
        int64_t,      /* Return type */
        FieldOp,      /* Parent class */
        getNOutChans, /* Name of function in C++ (must match Python name) */
    );
  }

  void setup0() const override {
    PYBIND11_OVERRIDE_PURE(
        void,    /* Return type */
        FieldOp, /* Parent class */
        setup0,  /* Name of function in C++ (must match Python name) */
    );
  }
};

/**
 * Add bindings for the pool op.
 **/
void bindPool(py::module &m);

} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_POOL_HPP
