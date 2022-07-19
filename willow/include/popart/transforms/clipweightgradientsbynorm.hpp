// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_CLIPWEIGHTGRADIENTSBYNORM_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_CLIPWEIGHTGRADIENTSBYNORM_HPP_

#include <cstddef>
#include <string>
#include <vector>
#include <popart/transforms/transform.hpp>

namespace popart {

class Op;
class Graph;

class ClipWeightGradientsByNorm : public Transform {
public:
  static std::size_t id();

  ClipWeightGradientsByNorm() : Transform() {}
  virtual ~ClipWeightGradientsByNorm() override {}
  virtual bool apply(Graph &graph) const final;
  virtual std::size_t getId() const final { return id(); }
  virtual std::string getName() const final {
    return "ClipWeightGradientsByNorm";
  }

  // Find and return all of the ops added by this transform when applying
  // gradient clipping. Each set of ops associated with clipping group will be
  // returned in a separate vector.
  static std::vector<std::vector<Op *>>
  findGradientClippingGroups(const Graph &graph);
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_CLIPWEIGHTGRADIENTSBYNORM_HPP_
