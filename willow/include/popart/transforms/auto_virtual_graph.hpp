// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_AUTO_VIRTUAL_GRAPH_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_AUTO_VIRTUAL_GRAPH_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <popart/transforms/transform.hpp>

#include "popart/names.hpp"

namespace popart {
class Graph;
class Op;

class Subgraph {
public:
  Subgraph(OpId op_id) : cost(0.f), candidates({op_id}), split_nodes({}) {}
  Subgraph(float c, OpId op_id)
      : cost(c), candidates({op_id}), split_nodes({}) {}

  float cost;
  std::set<OpId> candidates;
  std::map<float, OpId> split_nodes;

  std::set<OpId> final_splits;

  int64_t virtual_graph_id = 0;

  // Becomes true when the subgraph is first split over IPUs
  bool has_been_split = false;

  std::pair<bool, OpId> best_split(float split_cost);
};

class AutoVirtualGraph : public Transform {
public:
  static std::size_t id();

  AutoVirtualGraph() : Transform() {}
  ~AutoVirtualGraph() override {}

  bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "AutoVirtualGraph"; }

  float
  costFn(Op *op, bool training, float w_weights, float w_activations) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_AUTO_VIRTUAL_GRAPH_HPP_
