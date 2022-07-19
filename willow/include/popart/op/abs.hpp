// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ABS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ABS_HPP_

#include <map>
#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class AbsOp : public ElementWiseUnaryOp {
public:
  AbsOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class AbsGradOp : public Op {
public:
  AbsGradOp(const AbsOp &);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArgInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
  virtual float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ABS_HPP_
