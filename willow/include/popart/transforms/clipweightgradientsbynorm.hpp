// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CLIPBYNORMTRANSFORM_HPP
#define GUARD_NEURALNET_CLIPBYNORMTRANSFORM_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

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

#endif
