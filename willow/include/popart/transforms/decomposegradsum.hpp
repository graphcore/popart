// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_DECOMPOSEGRADSUM_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_DECOMPOSEGRADSUM_HPP_

#include <cstddef>
#include <string>
#include <vector>
#include <popart/transforms/decomposesum.hpp>

namespace popart {
class Graph;

class DecomposeGradSum : public DecomposeSum {
public:
  static std::size_t id();
  std::size_t getId() const override { return id(); }
  std::string getName() const override { return "DecomposeGradSum"; }

private:
  // Search graph for gradient Sum that will be decomposed
  // into an Add tree by this transform
  std::vector<Op *> getDecomposableSumOps(const Graph &) const override;
  void applyAddOpAttributes(Op *) const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_DECOMPOSEGRADSUM_HPP_
