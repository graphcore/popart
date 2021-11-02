// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_VARUPDATE_HPP
#define POPART__INTERNAL_IR_BINDINGS_VARUPDATE_HPP

#include "bindings/basicoptionals.hpp"
#include "bindings/op.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <popart/alias/aliasmodel.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/varupdate.hpp>

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
 */
template <class BaseOp = VarUpdateOp>
class PyVarUpdateOp : public PyOp<BaseOp> {
public:
  using PyOp<BaseOp>::PyOp;
  // See https://github.com/pybind/pybind11/issues/2185
  typedef std::map<InIndex, TensorId> InMap;

  InMap optimizerInputs() const override {
    PYBIND11_OVERRIDE_PURE(InMap,           /* Return type */
                           BaseOp,          /* Parent class */
                           optimizerInputs, /* Name of function in C++ (must
                                                   match Python name) */
    );
  }
  bool isOptimizerOp() const override {
    PYBIND11_OVERRIDE(bool,          /* Return type */
                      BaseOp,        /* Parent class */
                      isOptimizerOp, /* Name of function in C++ (must
                                              match Python name) */
    );
  }
  void growAliasModel(AliasModel &aliasmodel) const override {
    PYBIND11_OVERRIDE(
        void,           /* Return type */
        BaseOp,         /* Parent class */
        growAliasModel, /* Name of function in C++ (must match Python name) */
        aliasmodel);
  }
};

/**
 * Add bindings for the varupdate op.
 **/
void bindVarupdate(py::module &m);

} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_VARUPDATE_HPP
