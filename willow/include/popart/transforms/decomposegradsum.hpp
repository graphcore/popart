// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DECOMPOSEGRADSUM_HPP
#define GUARD_NEURALNET_DECOMPOSEGRADSUM_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

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
