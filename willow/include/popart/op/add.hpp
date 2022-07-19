// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ADD_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ADD_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>
#include <popart/op/reducesum.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"

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

class AddLhsInplaceOp : public ElementWiseBinaryInplaceLhsOp {
public:
  AddLhsInplaceOp(const OperatorIdentifier &_, const Op::Settings &_settings)
      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::AddLhsInplace,
                                      _settings) {}

  AddLhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::AddLhsInplace,
                                      _settings) {}
  std::unique_ptr<Op> clone() const final;
};

class AddRhsInplaceOp : public ElementWiseBinaryInplaceRhsOp {
public:
  AddRhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceRhsOp(Onnx::CustomOperators::AddRhsInplace,
                                      _settings) {}
  std::unique_ptr<Op> clone() const final;
};

class AddArg0GradOp : public ReduceSumOp {
public:
  AddArg0GradOp(const Op &, const std::vector<int64_t> &axes);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const override final;
  void setup() final;

  std::unique_ptr<Op> clone() const final;

private:
  Shape forward_op_arg_shape;
};

class AddArg1GradOp : public ReduceSumOp {
public:
  AddArg1GradOp(const Op &, const std::vector<int64_t> &axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;

  const std::map<int, int> &gradOutToNonGradIn() const override final;
  void setup() final;

  std::unique_ptr<Op> clone() const final;

private:
  Shape forward_op_arg_shape;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ADD_HPP_
