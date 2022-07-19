// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_NEGATE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_NEGATE_HPP_

#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

class NegateOp : public ElementWiseUnaryOp {
public:
  NegateOp(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class NegateGradOp : public NegateOp {
public:
  NegateGradOp(const NegateOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_NEGATE_HPP_
