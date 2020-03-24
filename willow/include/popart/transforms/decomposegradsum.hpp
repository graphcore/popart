// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DECOMPOSEGRADSUM_HPP
#define GUARD_NEURALNET_DECOMPOSEGRADSUM_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

// A class to encapsulate partial gradient tensors
// (the inputs to a gradient sum operation), in
// particular to aid with deciding the topology of
// the addition tree to which a set of GradPartials
// are the inputs
class GradPartial {
public:
  GradPartial(Tensor *, std::vector<Op *>);
  Tensor *t;
  std::vector<Op *> pathFromLoss;
  size_t pathLengthFromLoss() const;
  bool operator<(const GradPartial &) const;
};

class DecomposeGradSum : public Transform {
public:
  static std::size_t id();

  DecomposeGradSum() : Transform() {}
  virtual ~DecomposeGradSum() override {}
  virtual bool apply(Graph &graph) const final;
  virtual std::size_t getId() const final { return id(); }
  virtual std::string getName() const final { return "DecomposeGradSum"; }

private:
  // Search graph for gradient Sum that will be decomposed
  // into an Add tree by this transform
  std::vector<Op *> getDecomposableGradSumOps(const Graph &) const;
};

} // namespace popart

#endif
