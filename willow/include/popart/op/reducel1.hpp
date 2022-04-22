// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEL1_HPP
#define GUARD_NEURALNET_REDUCEL1_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/reduce.hpp>
#include <popart/vendored/optional.hpp> // IWYU pragma: keep

#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

class ReduceL1Op : public ReduceOp {
public:
  ReduceL1Op(const OperatorIdentifier &_opid,
             const nonstd::optional<std::vector<int64_t>> &axes,
             const int64_t keepdims,
             const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReduceL1GradOp : public ReduceGradOp {
public:
  ReduceL1GradOp(const ReduceL1Op &fwdOp, const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  static InIndex getFwdInInIndex() { return 1; }
};

} // namespace popart

#endif
