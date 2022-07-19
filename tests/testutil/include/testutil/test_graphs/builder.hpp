// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_BUILDER_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_BUILDER_HPP_

#include <map>
#include <memory>
#include <vector>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class Graph;
class Op;
} // namespace popart

namespace test_graphs {
namespace builder {

// TODO: Should the following Edge structs be encapsulated into a class heirachy
//       and then withEdges can take a single vector<Edge> and call virtual
//       methods on them to achieve the desired functionality?
//       Probably unnecessary complexity.

/**
 * \struct OpOpEdge
 *
 * \brief Describes an Op->Op edge.
 *
 * The tensor inbetween will be constructed implictly by `withEdges`.
 *
 * The output/input index at which the tensor is connected to the start/end op
 * is specified.
 *
 * The start op will be `setup` by `withEdges`.
 */
struct OpOpEdge {
  // The id of the op at the start of this edge.
  popart::OpId startId;

  // The output index at which the start op's output tensor will be connected.
  popart::OutIndex outIdx;

  // The id of the op at the end of this edge.
  popart::OpId endId;

  // The input index at which the start op's output tensor will be connected to
  // the end op.
  popart::InIndex inIdx;
};

/**
 * \struct TensorOpEdge
 *
 * \brief Describes a Tensor->Op edge.
 *
 * Conceptually, these are the inputs to the user's described graph. The start
 * tensor could be an input of the Popart graph; or it could be a tensor that
 * has no producer op because it is considered inherent to the graph (like how
 * Popart treats weights, for example); or it could be a tensor that is the
 * output of already existing op in the graph. It is up to the user.
 *
 * The input index at which the tensor is connected to the end op is specified.
 *
 * The tensor must already exist in the graph.
 */
struct TensorOpEdge {
  // The id of the tensor at the start of this edge, that will be connected to
  // the end op at `inIdx`.
  // Examples of how the user can create a new tensor (if needed):
  //     graph.addInput(tensorId, tensorInfo)
  //     graph.getTensors().addConstInit(appropiate, args, ...)
  //
  // Design Note: It is easier for the user to have to provide the tensor
  // themselves, rather than the complexity of us conditionally creating the
  // right kind of tensor based on their specification.
  popart::TensorId startId;

  // The id of the op that the tensor is an input of.
  popart::OpId endId;

  // The input index at which the tensor will be connected to the op.
  popart::InIndex inIdx;
};

/**
 * \struct OpTensorEdge
 *
 * \brief Describes an Op->Tensor edge.
 *
 * Conceptually, these are the outputs/"ends" of the user's described graph. The
 * end tensor could be be a graph output, or the input of an existing op in the
 * graph - it is up to the user.
 *
 * The end tensor can be an existing one or it can be requested to create a new
 * one.
 *
 * The out index at which the start op is connected to the tensor is specified.
 *
 * The start op will be `setup` by `withEdges`.
 */
struct OpTensorEdge {
  // The id of the Op at the start of this edge.
  popart::OpId startId;

  // The OutIndex of the start op at which the end tensor will be connected.
  popart::OutIndex outIdx;

  // The id of the tensor at the end of this edge. If not given, the tensor will
  // be created.
  nonstd::optional<popart::TensorId> endId;
};

/**
 * \brief Constructs in `graph` the topology described by the given ops and
 * edges.
 *
 * Edges can be made to existing ops as well as the new ops in upOps. All
 * tensors at the start or end of an edge must already exist.
 *
 * \param graph The graph to construct this topology in.
 * \param upOps New Ops to move into the graph. These Ops must be constructed,
 *              but not yet \ref popart::Op::setup, connected to the desired
 *              tensors, or moved into the graph. This will all be done by this
 *              function. Every op in upOps will be
 *              \ref popart::Op::setup(). Note, it is safe for the user to call
 *              \ref popart::Op::setup() again. Note, other (already existing)
 *              ops referenced in the edges will not be
 *              \ref popart::Op::setup().
 * \param opOps Op->Tensor->Op edges to construct. See `struct OpOpEdge`.
 * \param tenOps Tensor->Op edges to construct. These are effectively your graph
 *               inputs. See struct TensorOpEdge.
 * \param opTens Op->Tensor edges to construct. These are effectively your graph
 *               ouputs. See struct OpTensorEdge.
 * \param topoCons Additional topological constraints to insert in the graph's
 *                 \ref popart::Graph::topoCons.
 */
void withEdges(popart::Graph &graph,
               std::vector<std::unique_ptr<popart::Op>> &upOps,
               const std::vector<OpOpEdge> &opOps,
               const std::vector<TensorOpEdge> &tenOps,
               const std::vector<OpTensorEdge> &opTens,
               const std::multimap<popart::OpId, popart::OpId> &topoCons);

} // namespace builder
} // namespace test_graphs

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_BUILDER_HPP_
