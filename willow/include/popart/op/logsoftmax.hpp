// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOGSOFTMAX_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class LogSoftmaxOp : public ElementWiseUnaryOp {
public:
  LogSoftmaxOp(const OperatorIdentifier &_opid,
               int64_t axis,
               const Op::Settings &settings_);

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::unique_ptr<Op> clone() const final;

  int64_t getAxis() const;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

private:
  int64_t axis;
};

class LogSoftmaxInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  LogSoftmaxInplaceOp(const LogSoftmaxOp &);
  std::unique_ptr<Op> clone() const final;
  int64_t getAxis() const { return axis; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

private:
  int64_t axis;
};

class LogSoftmaxGradOp : public Op {
public:
  LogSoftmaxGradOp(const LogSoftmaxOp &);

  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;

  const std::map<int, int> &gradOutToNonGradIn() const final;

  void setup() final;

  int64_t getAxis() const { return axis; }

  static InIndex getGradProbsInIndex() { return 0; }

  static InIndex getActsInIndex() { return 1; }

  static OutIndex getOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  int64_t axis;
};

} // namespace popart

#endif
