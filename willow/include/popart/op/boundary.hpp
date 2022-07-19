// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_BOUNDARY_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_BOUNDARY_HPP_

#include <memory>
#include <popart/op.hpp>

#include "popart/operatoridentifier.hpp"

namespace popart {

// Dummy Op to signal boundaries
class BoundaryOp : public Op {
public:
  BoundaryOp(const Op::Settings &settings_)
      : Op(OperatorIdentifier("ai.graphcore", "Boundary", 1), settings_) {}
  std::unique_ptr<Op> clone() const override;
  void setup() final {}
  float getSubgraphValue() const final { return 0.0f; }
  bool isOutlineable() const override { return false; }
  bool hasSideEffect() const override { return true; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_BOUNDARY_HPP_
