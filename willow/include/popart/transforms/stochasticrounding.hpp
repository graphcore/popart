// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STOCHASTIC_ROUNDING_HPP
#define GUARD_NEURALNET_STOCHASTIC_ROUNDING_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

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

#endif
