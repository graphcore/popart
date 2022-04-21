// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_PARAMETERIZEDOPBINDER_HPP
#define POPART__INTERNAL_IR_BINDINGS_PARAMETERIZEDOPBINDER_HPP

#include <popart/basicoptionals.hpp>
#include <popart/graph.hpp>
#include <popart/op/custom/parameterizedop.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {
// Pybind11 helper functions.
// The goal is to standardize these as much as possible for all easily
// parametrize-able ops.

/**
 * @brief Make the parameterized op base Pybind11 class and methods.
 *
 * @tparam TParameterizedOp Parameterized op class.
 * @param m Pybind11 module
 * @param op_name Pybind11 op name to use.
 */
template <typename TParameterizedOp>
void makeParameterizedOpBindings(pybind11::module m, const char *op_name) {
  namespace py = pybind11;
  // A bit of template trickery to find the params type associated with the op.
  // This is basically getting the function pointer type of the member function
  // params.
  using TOpParamsRaw = typename std::result_of<decltype (
      &TParameterizedOp::params)(TParameterizedOp *)>::type;
  // This is removing const and reference from a type (i.e. from const int& to
  // int ).
  using TOpParams = std::remove_cv_t<std::remove_reference_t<TOpParamsRaw>>;

  using InMapType  = std::map<popart::InIndex, popart::TensorId>;
  using OutMapType = std::map<popart::OutIndex, popart::TensorId>;

  py::class_<TParameterizedOp, popart::Op, std::shared_ptr<TParameterizedOp>>(
      m, op_name)
      .def(py::init<const popart::OperatorIdentifier &,
                    const TOpParams &,
                    const popart::Op::Settings &>(),
           py::arg("opid"),
           py::arg("params"),
           py::arg("settings"))
      .def(py::init<const TOpParams &, const popart::Op::Settings &>(),
           py::arg("params"),
           py::arg("settings"))
      .def_property_readonly("params", &TParameterizedOp::params)
      .def_static("default_opid", &TParameterizedOp::defaultOperatorId)
      // Factory methods.
      .def_static("create_op_in_graph",
                  py::overload_cast<popart::Graph &,
                                    const InMapType &,
                                    const OutMapType &,
                                    const popart::OperatorIdentifier &,
                                    const TOpParams &,
                                    const popart::Op::Settings &>(
                      &TParameterizedOp::createOpInGraph),
                  py::arg("graph"),
                  py::arg("inputs"),
                  py::arg("outputs"),
                  py::arg("opid"),
                  py::arg("params"),
                  py::arg("settings"),
                  py::return_value_policy::reference)
      .def_static("create_op_in_graph",
                  py::overload_cast<popart::Graph &,
                                    const InMapType &,
                                    const OutMapType &,
                                    const TOpParams &,
                                    const popart::Op::Settings &>(
                      &TParameterizedOp::createOpInGraph),
                  py::arg("graph"),
                  py::arg("inputs"),
                  py::arg("outputs"),
                  py::arg("params"),
                  py::arg("settings"),
                  py::return_value_policy::reference);
}

} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_PARAMETERIZEDOPBINDER_HPP
