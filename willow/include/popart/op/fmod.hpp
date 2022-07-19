// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_FMOD_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_FMOD_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

// Implements modulo operator. The result has the same sign as the dividend.
class FmodOp : public ElementWiseBinaryOp {
public:
  FmodOp(const OperatorIdentifier &opId, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class FmodArg0GradOp : public ElementWiseBinaryArg0GradOp {
public:
  FmodArg0GradOp(const FmodOp &op, const std::vector<int64_t> &reductionAxes);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_FMOD_HPP_
