// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef GUARD_NEURALNET_IDENTITY_HPP
#define GUARD_NEURALNET_IDENTITY_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/loss.hpp>

namespace popart {

class IdentityOp : public ElementWiseUnaryOp {
public:
  IdentityOp(const OperatorIdentifier &_opid, const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  // For inplace support
  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
};

class IdentityInplaceOp : public IdentityOp {
public:
  IdentityInplaceOp(const OperatorIdentifier &_opid,
                    const Op::Settings &settings_);
  IdentityInplaceOp(const IdentityOp &concatOp);

  std::unique_ptr<Op> clone() const override;

  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
};

class IdentityGradOp : public IdentityOp {
public:
  IdentityGradOp(const IdentityOp &fwdOp);
  IdentityGradOp(const Settings &settings_);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

enum class ScaleByReplication { Yes, No };

class IdentityLossOp : public LossOp {
public:
  IdentityLossOp(
      const OperatorIdentifier &_opid,
      const ReductionType &reduction,
      const Op::Settings &settings_,
      const ScaleByReplication scaleByReplication = ScaleByReplication::Yes);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  bool canBeReplacedByIdentity() const override;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  ReductionType getReductionType() const { return reduction_type_; }
  ScaleByReplication getScaleByReplication() const {
    return scaleByReplication_;
  }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const ReductionType reduction_type_;
  // TODO: remove after T33184, as this is now redundant
  const ScaleByReplication scaleByReplication_;
};

class IdentityLossGradOp : public Op {

public:
  IdentityLossGradOp(const IdentityLossOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;
  bool canBeReplacedByIdentity() const override;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  ReductionType getReductionType() const { return reduction_type_; }
  ScaleByReplication getScaleByReplication() const {
    return scaleByReplication_;
  }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const ReductionType reduction_type_;
  // TODO: remove after T33184, as this is now redundant
  const ScaleByReplication scaleByReplication_;
  Shape outShape_;
};

} // namespace popart

#endif
