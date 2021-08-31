// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART__INTERNAL_IR_BINDINGS_GRAPH_HPP
#define POPART__INTERNAL_IR_BINDINGS_GRAPH_HPP

#include <pybind11/pybind11.h>

#include <popart/graph.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

/**
 * Add bindings for `popart::Graph` class to pybind module.
 **/
void bindGraph(py::module &m);

/**
 * Binds the create Op function to the graph class. This includes all the
 * createOp_<OpName> functions for all ops. See
 * python/popart._internal.ir/templates/graph.cpp.j2 which renders the
 * graph.cpp.gen file containing all these functions.
 * Rendered file will end up in
 * <build-dir>>/build/popart/python/popart._internal.ir/bindings/graph.gen.cpp
 *
 * \param g The pybind11 Graph class_ to bind the functions to.
 */
void bindCreateOpFunctionToGraphClass(py::class_<Graph> g);

/**
 * Binds the create connected Op function to the graph class. This includes all
 * the createConnectedOp_<OpName> functions for all ops. See
 * python/popart._internal.ir/templates/graph.cpp.j2 which renders the
 * graph.cpp.gen file containing all these functions.
 *
 * \param g The pybind11 Graph class_ to bind the functions to.
 */
void bindCreateConnectedOpFunctionToGraphClass(py::class_<Graph> g);

} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART__INTERNAL_IR_BINDINGS_GRAPH_HPP
