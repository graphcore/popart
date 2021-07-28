// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEMIN_HPP
#define GUARD_NEURALNET_REDUCEMIN_HPP

#include <popart/op.hpp>
#include <popart/op/reduce.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class ReduceMinOp : public ReduceOp {
public:
  ReduceMinOp(const OperatorIdentifier &_opid,
              const nonstd::optional<std::vector<int64_t>> &axes,
              const int64_t keepdims,
              const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReduceMinGradOp : public ReduceGradOp {
public:
  ReduceMinGradOp(const ReduceMinOp &fwdOp, const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  static InIndex getFwdInInIndex() { return 1; }
  static InIndex getFwdOutInIndex() { return 2; }
};

} // namespace popart

#endif
