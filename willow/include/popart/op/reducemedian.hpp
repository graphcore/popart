// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEMEDIAN_HPP
#define GUARD_NEURALNET_REDUCEMEDIAN_HPP

#include <popart/op/reduce.hpp>
#include <popart/opserialiser.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class ReduceMedianOp : public ReduceOp {
public:
  ReduceMedianOp(const OperatorIdentifier &opid,
                 const nonstd::optional<std::vector<int64_t>> &axes,
                 int64_t keepdims,
                 const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  void setup() override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static OutIndex getIndicesOutIndex() { return 1; }

  bool canBeReplacedByIdentity() const override;
};

class ReduceMedianGradOp : public ReduceGradOp {
public:
  ReduceMedianGradOp(const ReduceMedianOp &fwd_op, const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const override;

  static InIndex getIndicesInIndex() { return 1; }
};

} // namespace popart

#endif
