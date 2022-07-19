// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_TRANSPOSE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_TRANSPOSE_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>
#include <popart/region.hpp> // IWYU pragma: keep

#include "popart/names.hpp"

namespace popart {
class AliasModel;
class OpSerialiserBase;
struct OperatorIdentifier;

// Corresponds to the ONNX Transpose op
// for N-dimensional tensors.
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose

class TransposeBaseOp : public Op {
public:
  TransposeBaseOp(const OperatorIdentifier &_opid,
                  const Shape &perm_,
                  const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void setPerm(const Shape &value) { perm = value; }
  const Shape &getPerm() const { return perm; }

  std::vector<uint64_t> getPerm_u64() const;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  // Get the permutation required to reverse the Transpose operation
  Shape generateReversePermutation() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  bool canShard() const override { return true; }

  int getOutBatchAxis(OutIndex) const override;

  virtual void growAliasModel(AliasModel &) const override;

private:
  // the new permutation of the tensor axes
  Shape perm;
  void setDefaultPerm();
};

class TransposeOp : public TransposeBaseOp {
public:
  TransposeOp(const OperatorIdentifier &_opid,
              const Shape &perm_,
              const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool canBeReplacedByIdentity() const override;

  // For inplace support
  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

  bool isOutplaceViewChange() const override { return true; }
};

class TransposeInplaceOp : public TransposeBaseOp {
public:
  TransposeInplaceOp(const OperatorIdentifier &_opid,
                     const Shape &,
                     const Op::Settings &settings_);
  TransposeInplaceOp(const TransposeOp &);
  std::unique_ptr<Op> clone() const final;

  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }

  bool isInplaceViewChange() const override { return true; }
};

// TransposeGrad is a reverse transposition
class TransposeGradOp : public TransposeOp {
public:
  TransposeGradOp(const TransposeOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_TRANSPOSE_HPP_
