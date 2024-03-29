// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_REDUCEMIN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_REDUCEMIN_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/reduce.hpp>

#include "popart/names.hpp"

namespace nonstd {
namespace optional_lite {
template <typename T> class optional;
} // namespace optional_lite
} // namespace nonstd

namespace popart {
struct OperatorIdentifier;

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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_REDUCEMIN_HPP_
