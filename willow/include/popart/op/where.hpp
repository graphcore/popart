// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_WHERE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_WHERE_HPP_

#include <cstddef>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class AliasModel;
struct OperatorIdentifier;

class WhereOp : public Op {
public:
  WhereOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  // Inputs
  static InIndex conditionInIndex() { return 0; }
  static InIndex xInIndex() { return 1; }
  static InIndex yInIndex() { return 2; }

  // Ouputs
  static OutIndex outIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &aliasModel,
                     OperatorIdentifier opId) const override;

  void growAliasModel(AliasModel &m) const override;
};

class WhereLhsInplaceOp : public WhereOp {
public:
  WhereLhsInplaceOp(const WhereOp &op);

  std::unique_ptr<Op> clone() const override;

  view::Regions modifies(InIndex index) const final;
  view::Regions aliases(InIndex index, OutIndex) const final;
};

class WhereRhsInplaceOp : public WhereOp {
public:
  WhereRhsInplaceOp(const WhereOp &op);

  std::unique_ptr<Op> clone() const override;

  view::Regions modifies(InIndex index) const final;
  view::Regions aliases(InIndex index, OutIndex) const final;
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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_WHERE_HPP_
