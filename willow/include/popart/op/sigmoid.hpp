// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SIGMOID_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SIGMOID_HPP_

#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class SigmoidOp : public ElementWiseUnaryOp {
public:
  SigmoidOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class SigmoidInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  SigmoidInplaceOp(const SigmoidOp &);
  std::unique_ptr<Op> clone() const final;
};

class SigmoidGradOp : public Op {
public:
  SigmoidGradOp(const SigmoidOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdOutInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SIGMOID_HPP_
