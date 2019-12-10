#ifndef GUARD_NEURALNET_SOFTMAX_HPP
#define GUARD_NEURALNET_SOFTMAX_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class NllLoss;

class SoftmaxOp : public ElementWiseUnaryOp {
public:
  SoftmaxOp(const OperatorIdentifier &_opid,
            int64_t axis_,
            const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getAxis() const;
  void setAxis(int64_t);

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

private:
  int64_t axis;
};

class SoftmaxInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  SoftmaxInplaceOp(const SoftmaxOp &);
  std::unique_ptr<Op> clone() const final;
  int64_t getAxis() const { return axis; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

private:
  int64_t axis;
};

class SoftmaxGradOp : public Op {
public:
  SoftmaxGradOp(const SoftmaxOp &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  int gradProbsIn() const;
  int actsIn() const;
  int64_t getAxis() const;

  static InIndex getGradProbsInIndex() { return 0; }
  static InIndex getActsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  int64_t axis;
};

class SoftmaxGradDirectOp : public Op {
public:
  // where Op in this constructor must be a SoftmaxOp
  // where this is created by a merger between the Op
  // and an NllGradOp
  SoftmaxGradDirectOp(const NllLoss *, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  const NllLoss *nlll() const;
  bool hasNlllFwdOp() const;
  Op *nlllFwdOp() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool isOutlineable() const final { return false; }

private:
  const NllLoss *nllloss_;
};

class NlllWithSoftmaxGradDirectOp : public Op {
public:
  // where Op in this constructor must be a SoftmaxOp
  // where this is created by a merger between the Op
  // and an NllGradOp
  NlllWithSoftmaxGradDirectOp(const NllLoss *, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  const NllLoss *nlll() const;
  Op *nlllFwdOp() const;

  static InIndex getProbsInIndex() { return 0; }
  static InIndex getLabelInIndex() { return 1; }
  static InIndex getLossScalingInIndex() { return 2; }

  static OutIndex getLossOutIndex() { return 0; }
  static OutIndex getGradOutIndex() { return 1; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool isOutlineable() const final { return false; }

private:
  const NllLoss *nllloss_;
};

} // namespace popart

#endif
