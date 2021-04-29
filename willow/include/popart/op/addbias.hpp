// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADDBIAS_HPP
#define GUARD_NEURALNET_ADDBIAS_HPP

#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/reducesum.hpp>

namespace popart {

// A special purpose add operation used to add a bias to the output of a
// convolution operation.
class AddBiasOp : public Op {
public:
  AddBiasOp(const OperatorIdentifier &_opid, const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Current implementation places the data input at index 0, and the bias input
  // at index 1.
  static InIndex getDataInIndex() { return 0; }
  static InIndex getBiasInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const override;
  view::RegMap bwdRegMap(InIndex, OutIndex) const override;

  virtual void growAliaser(PoprithmsAliaser &) const override;

  void setProposal(poprithms::memory::inplace::Proposal &,
                   const PoprithmsAliaser &,
                   OperatorIdentifier) const override;
};

class AddBiasInplaceOp : public AddBiasOp {
public:
  AddBiasInplaceOp(const AddBiasOp &);

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;
};

// The gradient op for the data input of the add bias op.
// Based on the identity op
class AddBiasDataGradOp : public IdentityOp {
public:
  AddBiasDataGradOp(const AddBiasOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

// The gradient op for the bias input of the add bias op.
// Based on the reduce sum op.
class AddBiasBiasGradOp : public ReduceSumOp {
public:
  AddBiasBiasGradOp(const AddBiasOp &, const std::vector<int64_t> &axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
