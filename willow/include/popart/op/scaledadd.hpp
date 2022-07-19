// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SCALEDADD_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SCALEDADD_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {

class OpSerialiserBase;
class AliasModel;
struct OperatorIdentifier;

// z = a * x + b * y
class ScaledAddOp : public Op {
public:
  ScaledAddOp(const OperatorIdentifier &_opid,
              float scale_0_,
              float scale_1_,
              const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static InIndex getScale0InIndex() { return 2; }
  static InIndex getScale1InIndex() { return 3; }
  static OutIndex getOutIndex() { return 0; }

  float getScale0() const { return scale_0; }
  float getScale1() const { return scale_1; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>

  // Inplacing variants:
  // 1. If both scales are tensors, ScaledAdd can inplace on Arg0 or Arg1
  // 2. If both scales are constants, ScaledAdd can inplace on Arg0 or Arg1
  // 3. If scale_0 == 1.0 and scale_1 is a tensor, ScaledAdd can inplace on Arg0
  // 4. If scale_1 == 1.0 and scale_0 is a tensor, ScaledAdd can inplace on Arg1
  // 5. Otherwise, no inplacing possible
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  bool canShard() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  virtual void growAliasModel(AliasModel &) const override;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

protected:
  float scale_0;
  float scale_1;
};

class ScaledAddLhsInplaceOp : public ScaledAddOp {
public:
  ScaledAddLhsInplaceOp(float scale_0_,
                        float scale_1_,
                        const Op::Settings &settings_);
  ScaledAddLhsInplaceOp(const ScaledAddOp &);
  std::unique_ptr<Op> clone() const final;

  view::Regions modifies(InIndex) const final;
  view::Regions aliases(InIndex, OutIndex) const final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;
};

class ScaledAddRhsInplaceOp : public ScaledAddOp {
public:
  ScaledAddRhsInplaceOp(const ScaledAddOp &);
  std::unique_ptr<Op> clone() const final;

  view::Regions modifies(InIndex) const final;
  view::Regions aliases(InIndex, OutIndex) const final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SCALEDADD_HPP_
