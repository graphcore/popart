// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADD_HPP
#define GUARD_NEURALNET_ADD_HPP

#include <popart/op/elementwise.hpp>
#include <popart/op/reducesum.hpp>

namespace popart {

class AddArg0GradOp;
class AddArg1GradOp;

class AddOp : public ElementWiseNpBroadcastableBinaryWithGradOp<AddArg0GradOp,
                                                                AddArg1GradOp> {
public:
  AddOp(const OperatorIdentifier &_opid, const Op::Settings &_settings);
  std::unique_ptr<Op> clone() const override;

private:
  bool hasLhsInplaceVariant() const final;
  bool hasRhsInplaceVariant() const final;

  std::unique_ptr<Op> getLhsInplaceVariant() const final;
  std::unique_ptr<Op> getRhsInplaceVariant() const final;

  OperatorIdentifier getLhsOperatorIdentifier() const final;
  OperatorIdentifier getRhsOperatorIdentifier() const final;
};

class AddLhsInplaceOp : public ElementWiseBinaryInplaceLhsOp<AddLhsInplaceOp> {
public:
  AddLhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::AddLhsInplace,
                                      _settings) {}
};

class AddRhsInplaceOp : public ElementWiseBinaryInplaceRhsOp<AddRhsInplaceOp> {
public:
  AddRhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceRhsOp(Onnx::CustomOperators::AddRhsInplace,
                                      _settings) {}
};

class AddArg0GradOp : public ReduceSumOp {
public:
  AddArg0GradOp(const Op &, const std::vector<int64_t> &axes);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const override final;
  void setup() final;

private:
  TensorInfo forward_op_arg_info;
};

class AddArg1GradOp : public ReduceSumOp {
public:
  AddArg1GradOp(const Op &, const std::vector<int64_t> &axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;

  const std::map<int, int> &gradOutToNonGradIn() const override final;
  void setup() final;

private:
  TensorInfo forward_op_arg_info;
};

} // namespace popart

#endif
