// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_FMOD_HPP
#define GUARD_NEURALNET_FMOD_HPP

#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

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

#endif // !GUARD_NEURALNET_FMOD_HPP
