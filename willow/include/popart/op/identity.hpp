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

class IdentityLoss : public Loss {
public:
  // where input = output, both rank 0 (per sample)
  IdentityLoss(TensorId input, TensorId output, ReductionType rt);
  // There are no tensors streamed into this loss layer (unlike NLL for
  // example which has a label streamed in)
  std::vector<TensorId> getStreamTensorNames() const final;
  std::unique_ptr<Op> getOp(const Op::Settings &settings_) const final;
  const OperatorIdentifier &op_type() const final;
  TensorId getInputId() const;

  std::unique_ptr<Loss> clone() const final {
    return std::unique_ptr<Loss>(new IdentityLoss(*this));
  }
};

class IdentityLossOp : public LossOp {
public:
  IdentityLossOp(const OperatorIdentifier &_opid,
                 const IdentityLoss *identityloss,
                 const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  bool canBeReplacedByIdentity();
  const IdentityLoss *identityl() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const IdentityLoss *identityloss_;
};

class IdentityLossGradOp : public Op {

public:
  IdentityLossGradOp(const IdentityLossOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  const IdentityLoss *identityl() const;
  std::unique_ptr<Op> clone() const final;

  static InIndex getInIndex() { return 0; }
  static InIndex getGradInIndex() { return 1; }

  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const IdentityLoss *identityloss_;
};

} // namespace popart

#endif
