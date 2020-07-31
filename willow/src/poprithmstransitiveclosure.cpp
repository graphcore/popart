// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <unordered_set>
#include <vector>

#include <popart/graph.hpp>
#include <popart/poprithmstransitiveclosure.hpp>

#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace rithmic = poprithms::schedule::transitiveclosure;

namespace popart {

PoprithmsTransitiveClosure::PoprithmsTransitiveClosure(
    std::unordered_map<OpId, poprithms::schedule::transitiveclosure::OpId>
        toRithmicOpId,
    std::unordered_map<poprithms::schedule::transitiveclosure::OpId, OpId>
        toPopartOpId,
    poprithms::schedule::transitiveclosure::TransitiveClosure rithmicTC)
    : toRithmicOpId(std::move(toRithmicOpId)),
      toPopartOpId(std::move(toPopartOpId)), rithmicTC(std::move(rithmicTC)) {}

PoprithmsTransitiveClosure
PoprithmsTransitiveClosure::fromGraph(const Graph &g) {
  // We use a public factory function and a private memberwise constructor so we
  // can perform this logic once and construct the members only once - otherwise
  // if the actual constructor took the graph and built the members, it would
  // have to default construct the members first in the member initializer list,
  // then properly construct them again in the constructor body.

  const auto graphEdges = g.getEdgeMap();

  const auto nOps = graphEdges.size();

  // 1. Construct rithmic::OpId <-> OpId mappings. Note, rithmic::OpIds will be
  //    0...nOps-1.

  // Construct maps with nOps buckets, so operations should be O(1).
  std::unordered_map<OpId, rithmic::OpId> toRithmicOpId{nOps};
  std::unordered_map<rithmic::OpId, OpId> toPopartOpId{nOps};
  {
    rithmic::OpId nextRithmicOpId = 0;

    // Recall that `Graph::getEdgeMap` returns `a -> {}` if `a` has no
    // consumers, so we will not miss any OpIds by iterating over the map as so.
    for (const auto &consumersOfOp : graphEdges) {
      const auto opId        = consumersOfOp.first;
      const auto rithmicOpId = nextRithmicOpId++;

      toRithmicOpId.insert({opId, rithmicOpId});
      toPopartOpId.insert({rithmicOpId, opId});
    }
  }

  // 2. Construct the rithmic::Edges by converting graph edges on OpIds to a
  //    vector of vector of rithmic::OpIds, using the mapping `toRithmicOpId`.

  std::vector<std::vector<rithmic::OpId>> rithmicEdges(nOps);
  {
    for (const auto &consumersOfOp : graphEdges) {
      const auto opid_from = consumersOfOp.first;
      const auto &opids_to = consumersOfOp.second;

      const auto rithmic_from = toRithmicOpId[opid_from];

      for (const auto opid_to : opids_to) {
        const auto rithmic_to = toRithmicOpId[opid_to];
        rithmicEdges[rithmic_from].push_back(rithmic_to);
      }
    }
  }

  rithmic::TransitiveClosure rithmicTC{rithmicEdges};

  return {
      std::move(toRithmicOpId), std::move(toPopartOpId), std::move(rithmicTC)};
}

rithmic::OpId PoprithmsTransitiveClosure::rithmicOpId(const OpId opid) const {
  return toRithmicOpId.at(opid);
}

popart::OpId
PoprithmsTransitiveClosure::popartOpId(const rithmic::OpId opid) const {
  return toPopartOpId.at(opid);
}

std::size_t PoprithmsTransitiveClosure::numOps() const {
  const auto size = toRithmicOpId.size();

  assert(size == toPopartOpId.size());

  return size;
}

} // namespace popart
