// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SQRT_HPP
#define GUARD_NEURALNET_SQRT_HPP

#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

// y = sqrt(x)
class SqrtOp : public ElementWiseUnaryOp {
public:
  SqrtOp(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class SqrtGradOp : public Op {
public:
  SqrtGradOp(const SqrtOp &fwdOp);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdOutInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
