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
};

} // namespace popart

#endif
