// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_PRELU_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_PRELU_HPP_

#include <memory>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class PReluOp : public ElementWiseBinaryOp {
public:
  PReluOp(const OperatorIdentifier &opid_, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_PRELU_HPP_
