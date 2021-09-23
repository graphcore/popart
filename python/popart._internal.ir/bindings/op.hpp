// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_OP_HPP
#define POPART__INTERNAL_IR_BINDINGS_OP_HPP

#include <pybind11/pybind11.h>
#include <popart/op.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

/**
 * This is a trampoline class; discussed extensively here:
 * https://pybind11.readthedocs.io/en/stable/advanced/classes.html#classes
 *  As the base Op class has virtual and pure virtual methods, we must create
 * this in-between class that redirects virtual calls back to Python.
 *
 * \tparam BaseOp The op type. This is a template in case we have other Op
 * classes that require the trampoline. Defaults to Op.
 */
template <class BaseOp = Op> class PyOp : public BaseOp {
public:
  using BaseOp::BaseOp;

  // See discussion in https://github.com/pybind/pybind11/issues/673 for why
  // this is required
  // TODO: T41718 Derived op bindings may need to return a `shared_ptr` when
  // doing the clone operation.
  std::unique_ptr<Op> clone() const override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
        pybind11::get_overload(static_cast<const Op *>(this), "clone");
    auto o     = overload();
    auto shptr = pybind11::cast<std::shared_ptr<Op>>(o);
    return shptr->clone();
  }
  Op *clone_wrapper() const {
    PYBIND11_OVERLOAD_PURE(
        Op *,   /* Return type */
        BaseOp, /* Parent class */
        clone,  /* Name of function in C++ (must match Python name) */
    );
  }
  float getSubgraphValue() const override {
    PYBIND11_OVERRIDE_PURE(
        float,            /* Return type */
        BaseOp,           /* Parent class */
        getSubgraphValue, /* Name of function in C++ (must match Python name) */
    );
  }

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &opid) const override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload = pybind11::get_overload(
        static_cast<const Op *>(this), "getInplaceVariant");
    auto o     = overload(opid);
    auto shptr = pybind11::cast<std::shared_ptr<Op>>(o);
    return shptr->getInplaceVariant(opid);
  }

  std::shared_ptr<Op>
  getInplaceVariant_wrapper(const OperatorIdentifier &opid) const {
    PYBIND11_OVERLOAD(std::shared_ptr<Op>, /* Return type */
                      BaseOp,              /* Parent class */
                      getInplaceVariant,   /* Name of function in C++ (must
                                              match Python name) */
                      opid);
  }
};

/**
 * Add bindings for `popart::Op` class to pybind module.
 **/
void bindOp(py::module &m);

} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_OP_HPP
