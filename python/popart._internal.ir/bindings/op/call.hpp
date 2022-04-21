// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_CALL_HPP
#define POPART__INTERNAL_IR_BINDINGS_CALL_HPP

#include "bindings/basicoptionals.hpp"
#include "bindings/op.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

/**
 * Add bindings for the call op.
 **/
void bindCall(py::module &m);

} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_CALL_HPP