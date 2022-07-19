// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_NOP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_NOP_HPP_

#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

// The NopOp is functionally the same as the IdentityOp.
// The difference is that it will not be removed by patterns.
// It is meant as a debug operation to prevent patterns happening or prevent
// inplacing.
class NopOp : public ElementWiseUnaryOp {
public:
  NopOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  bool isOutplaceViewChange() const override { return true; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_NOP_HPP_
