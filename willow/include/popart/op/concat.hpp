// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONCAT_HPP
#define GUARD_NEURALNET_CONCAT_HPP

#include <popart/op.hpp>

namespace popart {

class ConcatOp : public Op {
public:
  ConcatOp(const OperatorIdentifier &_opid,
           int64_t axis_,
           const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  int64_t getAxis() const;

  // note that this is not final, ConcatInplaceOp overrides it
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;

  static InIndex getInIndex(InIndex index) { return index; }
  static OutIndex getOutIndex() { return 0; }

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;
  // "uses" is still the full input region
  // "aliases" is still the empty region
  // "modifies" is still the empty region

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  static Shape getOutputShape(int64_t axis,
                              const std::vector<const Shape *> inputs);

  bool canBeReplacedByIdentity() const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool canShard() const override { return true; }

  void growAliasModel(AliasModel &) const override;

  void setProposal(poprithms::memory::inplace::Proposal &,
                   const AliasModel &,
                   OperatorIdentifier) const override;

private:
  void validateAxis() const;

  int64_t axis = 0;

  // suppose input tensors have shapes,
  // 0: [2,5,3]
  // 1: [2,6,3]
  // 2: [2,1,3]
  // and axis = 1.
  // then the output tensor has shape [2,12,3], and
  // the regions in the output that inputs corresponds
  // to are,
  // 0: [0:2,  0:5,  0:3]
  // 1: [0:2,  5:11, 0:3]
  // 2: [0:2, 11:12, 0:3]
  // outOffests are where these regions start/end along "axis", so
  // in this case {0,5,11,12}
  int64_t getOutOffset(int64_t dim) const;

  void regMapPreChecks(InIndex inIndex) const;
};

// An inplace variant of the concat op
class ConcatInplaceOp : public ConcatOp {
public:
  ConcatInplaceOp(int64_t axis_, const Op::Settings &settings);
  ConcatInplaceOp(const ConcatOp &concatOp, int64_t axis_);

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {};
  }

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  // The whole of the used area is aliased. "modifies" is still empty
  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
};

class ConcatGradOp : public Op {
public:
  ConcatGradOp(const ConcatOp &op, InIndex input);
  ConcatGradOp(const ConcatInplaceOp &op, InIndex input);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  int64_t getAxis() const;
  int64_t getStart() const;
  int64_t getEnd() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }
  ReductionType getShardReductionType(OutIndex index) const override {
    return ReductionType::NoReduction;
  }

  void configureShardedOp(Op *const shardedOp,
                          const Settings *const settings_) const override;

protected:
  // An unsafe constructor that allows using any OperatorIdentifier
  ConcatGradOp(const OperatorIdentifier &_opid,
               const ConcatGradOp &concat_grad_op);

private:
  int64_t axis;
  int64_t start;
  int64_t end;
  InIndex fwdInput;

  TensorInfo gradInfo;
  std::map<int, int> gradOutToNonGradInInfo;
};

} // namespace popart

#endif
