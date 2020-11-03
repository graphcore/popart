// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBTRACT_HPP
#define GUARD_NEURALNET_SUBTRACT_HPP

#include <popart/op/elementwise.hpp>
#include <popart/op/reducesum.hpp>

namespace popart {

class SubtractArg0GradOp;
class SubtractArg1GradOp;

class SubtractOp
    : public ElementWiseNpBroadcastableBinaryWithGradOp<SubtractArg0GradOp,
                                                        SubtractArg1GradOp> {
public:
  SubtractOp(const OperatorIdentifier &_opid, const Op::Settings &_settings);
  std::unique_ptr<Op> clone() const final;
};

class SubtractArg0GradOp : public ReduceSumOp {
public:
  SubtractArg0GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

private:
  TensorInfo forward_op_arg_info;
};

class SubtractArg1GradOp
    : public ElementWiseBinaryArg1GradOp<SubtractArg1GradOp> {
public:
  SubtractArg1GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);

private:
  TensorInfo forward_op_arg_info;
};

} // namespace popart

#endif
