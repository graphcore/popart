// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCESUM_HPP
#define GUARD_NEURALNET_REDUCESUM_HPP

#include <popart/op.hpp>
#include <popart/op/reduce.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class ReduceSumOp : public ReduceOp {
public:
  ReduceSumOp(const OperatorIdentifier &_opid,
              const nonstd::optional<std::vector<int64_t>> &axes,
              const int64_t keepdims,
              const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReduceSumGradOp : public ReduceGradOp {
public:
  ReduceSumGradOp(const ReduceSumOp &fwdOp, const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
