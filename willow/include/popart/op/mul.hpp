// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MUL_HPP
#define GUARD_NEURALNET_MUL_HPP

#include <memory>
#include <vector>
#include <popart/names.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/reducesum.hpp>

namespace popart {

class MulArg0GradOp;
class MulArg1GradOp;

class MulOp : public ElementWiseNpBroadcastableBinaryWithGradOp<MulArg0GradOp,
                                                                MulArg1GradOp> {
public:
  MulOp(const OperatorIdentifier &_opid, const Op::Settings &_settings);
  std::unique_ptr<Op> clone() const final;
  static OperatorIdentifier getOpId(const Ir &ir);

private:
  bool hasLhsInplaceVariant() const final { return true; }
  bool hasRhsInplaceVariant() const final { return true; }

  std::unique_ptr<Op> getLhsInplaceVariant() const final;
  std::unique_ptr<Op> getRhsInplaceVariant() const final;

  OperatorIdentifier getLhsOperatorIdentifier() const final;
  OperatorIdentifier getRhsOperatorIdentifier() const final;
};

class MulLhsInplaceOp : public ElementWiseBinaryInplaceLhsOp<MulLhsInplaceOp> {
public:
  MulLhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::MulLhsInplace,
                                      _settings) {}
};

class MulRhsInplaceOp : public ElementWiseBinaryInplaceRhsOp<MulRhsInplaceOp> {
public:
  MulRhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceRhsOp(Onnx::CustomOperators::MulRhsInplace,
                                      _settings) {}
};

class MulArg0GradOp : public ElementWiseBinaryArg0GradOp<MulArg0GradOp> {
public:
  MulArg0GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
};

class MulArg1GradOp : public ElementWiseBinaryArg1GradOp<MulArg1GradOp> {
public:
  MulArg1GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
};

} // namespace popart

#endif
