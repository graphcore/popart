// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_PRUNE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_PRUNE_HPP_

#include <cstddef>
#include <set>
#include <string>
#include <vector>
#include <popart/transforms/transform.hpp>

namespace popart {

class Tensor;
class Op;
class Graph;

class PruneHelper {
public:
  PruneHelper(Graph *graph_) : graph(graph_) {}

  void setFront(std::vector<Tensor *> tensorFront_);

  void setRequired(std::set<Op *> required_);

  void analyze();

  const std::vector<Op *> &getOpsToDelete() const { return opsToDelete; }

  const std::vector<Tensor *> &getTensorsToDelete() const {
    return tensorsToDelete;
  }

  void deleteOps(const std::vector<Op *> &ops) const;

  void deleteTensors(const std::vector<Tensor *> &tensors) const;

private:
  Graph *graph;

  // Ops that can't be pruned
  std::set<Op *> required;

  // as we work backwards, we keep a
  // "front" of tensors,
  std::vector<Tensor *> tensorFront;

  // when a tensor enters the "front",
  // we record that it has been visited
  std::set<Tensor *> tensorsVisited;

  // ops \ required
  std::vector<Op *> opsToDelete;

  // all outputs of opsToDelete
  std::vector<Tensor *> tensorsToDelete;
};

class Prune : public Transform {
public:
  static std::size_t id();

  Prune() : Transform() {}
  ~Prune() override {}

  bool apply(Graph &graph) const override;

  std::size_t getId() const override { return id(); }

  std::string getName() const override { return "Prune"; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_PRUNE_HPP_
