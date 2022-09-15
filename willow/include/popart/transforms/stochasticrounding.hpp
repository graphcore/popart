// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_STOCHASTICROUNDING_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_STOCHASTICROUNDING_HPP_

#include <cstddef>
#include <string>
#include <popart/transforms/transform.hpp>

namespace popart {
class Graph;

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
