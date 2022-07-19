// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ABORT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ABORT_HPP_

#include <memory>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

class AbortOp : public Op {
public:
  AbortOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;

  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  // Index for tensor to assert on
  static InIndex getInIndex() { return 0; }

  bool hasSideEffect() const override { return true; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ABORT_HPP_
