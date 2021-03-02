// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAPHUTILS_HPP
#define GUARD_NEURALNET_GRAPHUTILS_HPP

#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

namespace popart {

namespace graphutils {

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
 * Traverse a graph starting from tensors
 *
 * \param tensors The \a tensors to start from
 * \param visitor The \a visitor function to call. The \a visitor should return
 * true if further tensors along that path should be explored. \param filter The
 * \a filter function to call. The \a filter should return true if Tensor A -> B
 * through Op is a path that should be traversed. \param traversalType How a
 * graph should be traversed \param visitType When the visitor should be called
 * \param traversalDirection Which directions should be traversed
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

} // namespace graphutils
} // namespace popart

#endif
