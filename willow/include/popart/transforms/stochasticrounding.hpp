// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_STOCHASTICROUNDING_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_STOCHASTICROUNDING_HPP_

#include <cstddef>
#include <string>
#include <popart/transforms/transform.hpp>

namespace popart {
class Graph;

// Set Op::Settings::stochasticRoundingMethod for all ops. Note that currently
// this transform sets the method to `StochasticRoundingMethod::IdenticalSeeds`.
//
// NOTE: The result of this is that all Ops use the same RNG state and, as a
// result, weights on different replicas may diverge. This matches historic
// behaviour, but is not where we want to eventually end up. With T42299, this
// transform will change and set the attribute to improve stochastic rounding
// behaviour.

class StochasticRounding : public Transform {
public:
  static std::size_t id();

  StochasticRounding() : Transform() {}
  virtual ~StochasticRounding() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "StochasticRounding"; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_STOCHASTICROUNDING_HPP_
