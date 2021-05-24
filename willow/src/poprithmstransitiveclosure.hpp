// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_TRANSITIVE_CLOSURE_HPP
#define GUARD_TRANSITIVE_CLOSURE_HPP

#include <unordered_map>

#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>
#include <popart/graph.hpp>

namespace popart {

/**
 * This is an adapter class for constructing a CONST
 * `poprithms::schedule::transitiveclosure::TransitiveClosure` from a
 * `popart::Graph`.
 *
 * The user is provided full access to the underlying poprithms transitive
 * closure API as well as the bidrectional mapping between `popart::OpId` and
 * `poprithms::schedule::transitiveclosure::OpId`.
 *
 * This class also provides the following guarantees on the nature of the
 * `poprithms::schedule::transitiveclosure::OpId`s created:
 *
 *   1. The total ordering of `popart::OpId`s will be preserved.
 *   2. The `poprithms::schedule::transitiveclosure::OpId`s will be contiguous
 *      natural numbers.
 *   3. The `poprithms::schedule::transitiveclosure::OpId`s will start at 0.
 *
 * Together, this means that the `poprithms::schedule::transitiveclosure::OpId`s
 * will be 0..N, where N is the number of ops in the graph, and that the
 * `poprithms::schedule::transitiveclosure::OpId` will be the corresponding
 * OpId's position in their ordering.
 *
 * For example, for a graph containing OpIds 10, 20, 50, 70, 71; the mapping
 * to `poprithms::schedule::transitiveclosure::OpId`s will be:
 *
 *   10 -> 0
 *   20 -> 1
 *   50 -> 2
 *   70 -> 3
 *   71 -> 4
 */
class PoprithmsTransitiveClosure {
public:
  // Factory func to create from graph.
  static PoprithmsTransitiveClosure fromGraph(const Graph &g);

  // Has all default copy/move/destruct.
  // TODO: Should there be a default constructor? Should poprithms TC have one?

  // Accessors for underlying poprithms TransitiveClosure functionality.

  const poprithms::schedule::transitiveclosure::TransitiveClosure *
  operator->() const {
    return &rithmicTC;
  }

  const poprithms::schedule::transitiveclosure::TransitiveClosure &
  operator*() const {
    return rithmicTC;
  }

  const poprithms::schedule::transitiveclosure::TransitiveClosure &get() const {
    return rithmicTC;
  }

  // Throw if OpId not found.
  poprithms::schedule::transitiveclosure::OpId rithmicOpId(const OpId) const;
  OpId popartOpId(const poprithms::schedule::transitiveclosure::OpId) const;

  std::size_t numOps() const;

private:
  std::unordered_map<OpId, poprithms::schedule::transitiveclosure::OpId>
      toRithmicOpId;
  std::unordered_map<poprithms::schedule::transitiveclosure::OpId, OpId>
      toPopartOpId;
  poprithms::schedule::transitiveclosure::TransitiveClosure rithmicTC;

  // Private, memberwise, forwarding constructor.
  PoprithmsTransitiveClosure(
      std::unordered_map<OpId, poprithms::schedule::transitiveclosure::OpId>
          toRithmicOpId,
      std::unordered_map<poprithms::schedule::transitiveclosure::OpId, OpId>
          toPopartOpId,
      poprithms::schedule::transitiveclosure::TransitiveClosure rithmicTC);
};

} // namespace popart

#endif
