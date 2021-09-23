// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_MANUALBINDOPS_HPP
#define POPART__INTERNAL_IR_BINDINGS_MANUALBINDOPS_HPP

#include <pybind11/pybind11.h>

#include <popart/graph.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

/**
 * These functions bind the Graph::createOp<OpName> functions to the graph class
 * in python internal _ir module. Most of these are created by template as part
 * of scripts/gen_op_bindings.py. However a few are difficult and require manual
 * intervention to bind. These functions bind the manually defined createOp
 * functions.
 *
 * \param g The python graph class to bind to.
 */
void bindManualCreateOpFunctionToGraphClass(py::class_<Graph> g);

/**
 * Same as above but binds the Graph::createConnectedOp<OpName> function.
 *
 * \param g
 */
void bindManualCreateConnectedOpFunctionToGraphClass(py::class_<Graph> g);

} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_MANUALBINDOPS_HPP
