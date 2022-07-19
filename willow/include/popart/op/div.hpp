// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_DIV_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_DIV_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {

class DivArg0GradOp;
class DivArg1GradOp;
struct OperatorIdentifier;

class DivOp : public ElementWiseNpBroadcastableBinaryWithGradOp<DivArg0GradOp,
                                                                DivArg1GradOp> {
public:
  DivOp(const OperatorIdentifier &_opid, const Op::Settings &_settings);
  std::unique_ptr<Op> clone() const final;
};

// gradOut / arg_1
class DivArg0GradOp : public ElementWiseBinaryArg0GradOp {
public:
  DivArg0GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
  std::unique_ptr<Op> clone() const final;
};

// - (gradOut * arg_0) / arg_1^2
class DivArg1GradOp : public ElementWiseBinaryArg1GradOp {
public:
  DivArg1GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_DIV_HPP_
