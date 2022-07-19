// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SQUARE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SQUARE_HPP_

#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class SquareOp : public ElementWiseUnaryOp {
public:
  SquareOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SQUARE_HPP_
