// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADD_HPP
#define GUARD_NEURALNET_ADD_HPP

#include <popart/op/elementwise.hpp>
#include <popart/op/reducesum.hpp>

namespace popart {

class AddOp : public ElementWiseBinaryOp {
public:
  AddOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

private:
  bool hasLhsInplaceVariant() const final;
  bool hasRhsInplaceVariant() const final;

  std::unique_ptr<Op> getLhsInplaceVariant() const final;
  std::unique_ptr<Op> getRhsInplaceVariant() const final;

  OperatorIdentifier getLhsOperatorIdentifier() const final;
  OperatorIdentifier getRhsOperatorIdentifier() const final;
};

class AddLhsInplaceOp : public ElementWiseBinaryInplaceLhsOp {
public:
  AddLhsInplaceOp(const AddOp &addOp);
  AddLhsInplaceOp(const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
};

class AddRhsInplaceOp : public ElementWiseBinaryInplaceRhsOp {
public:
  AddRhsInplaceOp(const AddOp &addOp);
  AddRhsInplaceOp(const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
};

class AddArg0GradOp : public ReduceSumOp {
public:
  AddArg0GradOp(const AddOp &, const std::vector<int64_t> &axes);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  TensorInfo forward_op_arg_info;
};

class AddArg1GradOp : public ReduceSumOp {
public:
  AddArg1GradOp(const AddOp &, const std::vector<int64_t> &axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  TensorInfo forward_op_arg_info;
};

} // namespace popart

#endif
