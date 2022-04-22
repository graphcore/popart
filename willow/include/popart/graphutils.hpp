// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAPHUTILS_HPP
#define GUARD_NEURALNET_GRAPHUTILS_HPP

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "popart/names.hpp"

namespace popart {
class Graph;
class Op;
class Tensor;
struct POpCmp;

namespace graphutils {

/**
 * CallStack representation
 */
using CallStack          = std::vector<Op *>;
using TensorAndCallStack = std::pair<Tensor *, CallStack>;

/**
 * Enum type that specifies how a graph should be traversed
 */
enum class TraversalType {
  /// Depth first (DFS)
  DepthFirst = 0,
  /// Breadth first (BFS)
  BreadthFirst
};

/**
 * Enum type that specifies when the visitor should be called
 */
enum class VisitType {
  /// Call visitor before enqueueing the next tensors to visit
  Pre = 0,
  /// Call visitor after enqueueing the next tensors to visit
  Post
};

/**
 * Enum type that specifies which directions of the graph should be traversed
 */
enum class TraversalDirection {
  /// Visit producers, consumers, graph inputs, graph outputs (forward first)
  ForwardBackward = 0,
  /// Visit producers, consumers, graph inputs, graph outputs (backward first)
  BackwardForward,
  /// Visit consumers and graph inputs (into subgraphs), graph outputs (out of
  /// subgraphs)
  Forward,
  /// Visit producers and graph inputs (out of subgraphs), graph outputs (into
  /// subgraphs)
  Backward
};

/**
 * Enum type that specifies how to traverse callsites
 */
enum class TraverseCallSites {
  /// Visit the only the producers/consumers of the current callstack when
  /// traversing out of subgraphs.
  Current = 0,
  /// Visit all callsites of a Graph when traversing out of subgraphs
  All
};

/**
 * Traverse a graph starting from tensors (with call stack)
 *
 * \param tensors            The \a tensors (and call stack) to start from
 * \param visitor            The \a visitor function to call. The \a visitor
 *                           should return true if further tensors along that
 *                           path should be explored.
 * \param filter             The \a filter function to call. The \a filter
 *                           should return true if Tensor A -> B through Op is a
 *                           path that should be traversed.
 * \param traversal          How a graph should be traversed.
 * \param visitType          When the visitor should be called.
 * \param traversalDirection Which directions should be traversed.
 * \param traverseCallSites  How to traverse out of subgraphs.
 */
void traverse(std::vector<TensorAndCallStack> tensors,
              std::function<bool(Tensor *)> visitor,
              std::function<bool(Op *, Tensor *, Tensor *)> filter,
              TraversalType traversalType,
              VisitType visitType,
              TraversalDirection traversalDirection,
              TraverseCallSites traverseCallSites);

/**
 * Traverse a graph starting from tensors
 *
 * \param tensors            The \a tensors to start from
 * \param visitor            The \a visitor function to call. The \a visitor
 *                           should return true if further tensors along that
 *                           path should be explored.
 * \param filter             The \a filter function to call. The \a filter
 *                           should return true if Tensor A -> B through Op is a
 *                           path that should be traversed.
 * \param traversal          How a graph should be traversed.
 * \param visitType          When the visitor should be called.
 * \param traversalDirection Which directions should be traversed.
 * \param traverseCallSites  How to traverse out of subgraphs.
 */
void traverse(std::vector<Tensor *> tensors,
              std::function<bool(Tensor *)> visitor,
              std::function<bool(Op *, Tensor *, Tensor *)> filter,
              TraversalType traversalType,
              VisitType visitType,
              TraversalDirection traversalDirection,
              TraverseCallSites traverseCallSites);

/**
 * Traverse a graph starting from tensors
 *
 * \param tensors            The \a tensors to start from
 * \param visitor            The \a visitor function to call. The \a visitor
 *                           should return true if further tensors along that
 *                           path should be explored.
 * \param filter             The \a filter function to call. The \a filter
 *                           should return true if Tensor A -> B through Op is a
 *                           path that should be traversed.
 * \param traversal          How a graph should be traversed.
 * \param visitType          When the visitor should be called.
 * \param traversalDirection Which directions should be traversed.
 */
void traverse(std::vector<Tensor *> tensors,
              std::function<bool(Tensor *)> visitor,
              std::function<bool(Op *, Tensor *, Tensor *)> filter,
              TraversalType traversalType,
              VisitType visitType,
              TraversalDirection traversalDirection);

/**
 * Traverse a graph starting from tensors (BFS, pre, bidirectional)
 *
 * \param tensors The \a tensors to start from
 * \param visitor The \a visitor function to call. The \a visitor should return
 * true if further tensors along that path should be explored.
 */
void traverseBreadthFirst(std::vector<Tensor *> tensors,
                          std::function<bool(Tensor *)> visitor);

/**
 * Traverse a graph starting from tensors (DFS, pre, bidirectional)
 *
 * \param tensors The \a tensors to start from
 * \param visitor The \a visitor function to call. The \a visitor should return
 * true if further tensors along that path should be explored.
 */
void traverseDepthFirst(std::vector<Tensor *> tensors,
                        std::function<bool(Tensor *)> visitor);

/**
 * Return the root tensors (tensors that are not graph inputs and without
 * producers (except InitOp)) starting from \a tensors. (DFS, pre, backward)
 *
 * \param tensor The \a tensor to start from
 */
std::vector<Tensor *> rootTensors(Tensor *tensor);

/**
 * Return pointers to Ops, together with which
 * of the other Ops have to occur
 * (by topo cons and input/output relations)
 * before the current Op
 */
std::map<Op *, std::set<Op *, POpCmp>, POpCmp>
getOpsWithBefores(const std::set<Op *, POpCmp> &ops);
std::map<Op *, std::set<Op *, POpCmp>, POpCmp>
getOpsWithBefores(const std::vector<Op *> &ops);

/**
 * Walk back from the current op and ensure we do not encounter
 * any of the ops in the vector of potential dependency ops
 * \param the op from which to walk back
 * \param opSchedule the schedule of ops which is used for the traversal
 * \param potentialDependencyOps a list of ops against which to check for
 *        data dependency
 * \return true, if there is a data dependency between op and any of
 *        the potential dependency ops
 */
bool hasDataDependency(Op *const op,
                       const std::vector<Op *> &opSchedule,
                       const std::vector<Op *> &potentialDependencyOps);

/**
 * Enum type that specifies the type of edge between operations
 */
enum class EdgeType {
  /// Expect a tensor edge between the operations
  Tensor = 0,
  /// Expect a topological constraint edge between operations
  TopoCon
};

class Edge {
public:
  Edge()
      : fromIndex(-1), toIndex(-1), out(-1), in(-1),
        edgeType(EdgeType::Tensor) {}
  Edge(int fromIndex_, int toIndex_)
      : fromIndex(fromIndex_), toIndex(toIndex_), out(-1), in(-1),
        edgeType(EdgeType::Tensor) {}
  Edge(int fromIndex_, int toIndex_, OutIndex out_, InIndex in_)
      : fromIndex(fromIndex_), toIndex(toIndex_), out(out_), in(in_),
        edgeType(EdgeType::Tensor) {}
  Edge(int fromIndex_, int toIndex_, EdgeType edgeType_)
      : fromIndex(fromIndex_), toIndex(toIndex_), out(-1), in(-1),
        edgeType(edgeType_) {}

  int getFrom() const { return fromIndex; }
  int getTo() const { return toIndex; }

  OutIndex getOut() const { return out; }

  InIndex getIn() const { return in; }

  EdgeType getEdgeType() const { return edgeType; }

private:
  // The predicate vector index
  int fromIndex;
  // The predicate vector index
  int toIndex;
  // The output index on the from-Op to follow (optional, defaults to -1 (follow
  // all indices))
  OutIndex out;
  // The input index on the to-Op to check (optional, defaults to -1 (check all
  // indices))
  InIndex in;
  // If the edge is a topological constraint rather than a data path
  EdgeType edgeType;
};

bool operator<(const Edge &a, const Edge &b);

using OpPred    = std::function<bool(const Op *op)>;
using OpPreds   = std::vector<OpPred>;
using OpPredMap = std::map<size_t, OpPred>;
using Edges     = std::set<Edge>;

/**
 * Returns Ops matching the \a preds connected by directed \a edges
 * \param preds Predicate functions that match Ops
 * \param edges Connectivity matrix between predicated Ops, where the fromIndex
 *              and toIndex of the \a edges correspond to the indices in the
 *              \a preds vector.
 * \return vector of all matches
 */
std::vector<std::vector<Op *>>
findMatchingOps(Graph &graph, const OpPreds &preds, const Edges &edges);

/**
 * Returns Ops matching the \a preds connected by directed \a edges
 * \param preds Predicate functions that match Ops
 * \param edges Connectivity matrix between predicated Ops, where the fromIndex
 *              and toIndex of the \a edges correspond to the indices in the
 *              \a preds vector.
 * \return vector of all matches
 */
std::vector<std::vector<Op *>>
findMatchingOps(Graph &graph, const OpPredMap &preds, const Edges &edges);

/**
 * Enum categorizing operations by their relation to the final loss.
 * Can be used to optimize recomputation.
 *
 * The operations are classified into four types by their position relative
 * to the final loss:
 *
 * ToLoss
 * |     \
 * |      FromToLoss
 * Loss
 * |      ToFromLoss
 * |     /
 * FromLoss
 *
 */
enum class OpFinalLossRelation {
  /// The operation leads to the final loss or is the final loss (i.e. is
  /// upstream as seen from the loss)
  /// (PathToLoss::Yes && PathFromLoss::{Yes, No, Undefined})
  ToLoss = 0,
  /// The operation comes from the final loss (i.e. is downstream as seen from
  /// the loss)
  /// (PathToLoss::{No, Undefined} && PathFromLoss::Yes)
  FromLoss,
  /// The operation is upstream of a FromLoss consumer (by data), or downstream
  /// of a FromLoss operation (by topocon)
  ToFromLoss,
  /// The operation is downstream of a ToLoss producer (by data), or upstream
  /// of a ToLoss operation (by topocon)
  FromToLoss,
};

std::ostream &operator<<(std::ostream &os, const OpFinalLossRelation &oprel);

/**
 * Categorizes all operations according to their position with relation
 * to the final loss, based on toLoss/fromLoss annotations, data dependencies
 * and schedule constraints.
 * \param graph Graph with operations to classify
 * \return      Relation map for each Op in the graph
 */
std::map<Op *, OpFinalLossRelation, POpCmp>
getOpFinalLossRelations(Graph &graph);

} // namespace graphutils
} // namespace popart

#endif
