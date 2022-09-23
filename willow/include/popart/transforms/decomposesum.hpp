// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_DECOMPOSESUM_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_DECOMPOSESUM_HPP_

#include <cstddef>
#include <string>
#include <vector>
#include <popart/transforms/transform.hpp>

namespace popart {
class Graph;
class Op;

class DecomposeSum : public Transform {
public:
  static std::size_t id();

  DecomposeSum() : Transform() {}
  ~DecomposeSum() override {}
  bool apply(Graph &graph) const override;
  std::size_t getId() const override { return id(); }
  std::string getName() const override { return "DecomposeSum"; }

private:
  // Search graph for SumOps with >2 inputs
  virtual std::vector<Op *> getDecomposableSumOps(const Graph &) const;
  virtual void applyAddOpAttributes(Op *) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_DECOMPOSESUM_HPP_
