// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DIV_HPP
#define GUARD_NEURALNET_DIV_HPP

#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

class DivArg0GradOp;
class DivArg1GradOp;

class DivOp : public ElementWiseNpBroadcastableBinaryWithGradOp<DivArg0GradOp,
                                                                DivArg1GradOp> {
public:
  DivOp(const OperatorIdentifier &_opid, const Op::Settings &_settings);
  std::unique_ptr<Op> clone() const final;
};

// gradOut / arg_1
class DivArg0GradOp : public ElementWiseBinaryArg0GradOp<DivArg0GradOp> {
public:
  DivArg0GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
};

// - (gradOut * arg_0) / arg_1^2
class DivArg1GradOp : public ElementWiseBinaryArg1GradOp<DivArg1GradOp> {
public:
  DivArg1GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
};

} // namespace popart

#endif
