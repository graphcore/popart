// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_RELU_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_RELU_HPP_

#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class ReluOp : public ElementWiseUnaryOp {
public:
  ReluOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class ReluInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  ReluInplaceOp(const ReluOp &);
  ReluInplaceOp(const Op::Settings &opSettings);
  std::unique_ptr<Op> clone() const final;
};

// takes output of ReluOp as input and not the input of ReluOp
// to determine where gradients become zero. It might be better
// (depending in what can be in-placed) to rather take the input
// of ReluOp in to do this (or a boolean tensor).
class ReluGradOp : public Op {
public:
  ReluGradOp(const ReluOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  // The index at which the output of the Relu (the "relud" tensor)
  // is an input to this ReluGradOp
  static InIndex getReludInIndex() { return 1; }

  // The index at which the gradient of the output of
  // the Relu is an input to this ReluGradOp
  static InIndex getGradReludInIndex() { return 0; }

  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_RELU_HPP_
