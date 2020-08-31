// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BOUNDARY_HPP
#define GUARD_NEURALNET_BOUNDARY_HPP

#include <popart/op.hpp>

namespace popart {

// Dummy Op to signal boundaries
class BoundaryOp : public Op {
public:
  BoundaryOp(const Op::Settings &settings_)
      : Op(OperatorIdentifier("", "", 0), settings_) {}
  std::unique_ptr<Op> clone() const override {
    return std::make_unique<BoundaryOp>(*this);
  }
  void setup() final {}
  float getSubgraphValue() const final { return 0.0f; }
  bool isOutlineable() const override { return false; }
  bool hasSideEffect() const override { return true; }
};

} // namespace popart

#endif
