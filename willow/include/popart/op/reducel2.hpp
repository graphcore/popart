// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_REDUCEL2_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_REDUCEL2_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/reduce.hpp>
#include <popart/vendored/optional.hpp> // IWYU pragma: keep

#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

class ReduceL2Op : public ReduceOp {
public:
  ReduceL2Op(const OperatorIdentifier &_opid,
             const nonstd::optional<std::vector<int64_t>> &axes,
             const int64_t keepdims,
             const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReduceL2GradOp : public ReduceGradOp {
public:
  ReduceL2GradOp(const ReduceL2Op &fwdOp, const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  static InIndex getFwdInInIndex() { return 1; }
  static InIndex getFwdOutInIndex() { return 2; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_REDUCEL2_HPP_
