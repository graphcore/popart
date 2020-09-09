// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_WHERE_HPP
#define GUARD_NEURALNET_WHERE_HPP

#include <popart/op.hpp>

namespace popart {

class WhereOp : public Op {
public:
  WhereOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() final;

  // Inputs
  static InIndex conditionInIndex() { return 0; }
  static InIndex xInIndex() { return 1; }
  static InIndex yInIndex() { return 2; }

  // Ouputs
  static OutIndex outIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class WhereXGradOp : public Op {
public:
  WhereXGradOp(const WhereOp &op);

  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex fwdConditionInIndex() { return 0; }
  static InIndex outGradInIndex() { return 1; }
  static OutIndex outIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::vector<size_t> getFwdInShape() const;

private:
  const TensorInfo fwdOpXInInfo;
};

class WhereYGradOp : public Op {
public:
  WhereYGradOp(const WhereOp &op);

  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex fwdConditionInIndex() { return 0; }
  static InIndex outGradInIndex() { return 1; }
  static OutIndex outIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::vector<size_t> getFwdInShape() const;

private:
  const TensorInfo fwdOpYInInfo;
};

} // namespace popart

#endif
