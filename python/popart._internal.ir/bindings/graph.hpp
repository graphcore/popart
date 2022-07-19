// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_GRAPH_HPP_
#define POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_GRAPH_HPP_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace popart {
class Graph;

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
 * graph.gen.cpp file containing all these functions.
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
 * graph.gen.cpp file containing all these functions.
 *
 * \param g The pybind11 Graph class_ to bind the functions to.
 */
void bindCreateConnectedOpFunctionToGraphClass(py::class_<Graph> g);

} // namespace ir
} // namespace _internal
} // namespace popart

#endif // POPART_PYTHON_POPART__INTERNAL_IR_BINDINGS_GRAPH_HPP_
