// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSPOSE_HPP
#define GUARD_NEURALNET_TRANSPOSE_HPP

#include <popart/op.hpp>

namespace popart {

// Corresponds to the ONNX Transpose op
// for N-dimensional tensors.
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose

class TransposeBaseOp : public Op {
public:
  TransposeBaseOp(const OperatorIdentifier &_opid,
                  const Shape &perm_,
                  const Op::Settings &settings_);

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void setPerm(const Shape &value) { perm = value; }
  const Shape &getPerm() const { return perm; }

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  // Get the permutation required to reverse the Transpose operation
  Shape generateReversePermutation() const;

  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

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

  bool canBeReplacedByIdentity() override;

  // For inplace support
  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
};

class TransposeInplaceOp : public TransposeBaseOp {
public:
  TransposeInplaceOp(const OperatorIdentifier &_opid,
                     const Shape &,
                     const Op::Settings &settings_);
  TransposeInplaceOp(const TransposeOp &);
  std::unique_ptr<Op> clone() const final;

  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
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

#endif
