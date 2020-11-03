// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POW_HPP
#define GUARD_NEURALNET_POW_HPP

#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

class PowArg0GradOp;
class PowArg1GradOp;

// arg_0 / arg_1
class PowOp : public ElementWiseNpBroadcastableBinaryWithGradOp<PowArg0GradOp,
                                                                PowArg1GradOp> {
public:
  PowOp(const OperatorIdentifier &_opid, const Op::Settings &_settings);
  std::unique_ptr<Op> clone() const final;

  static OperatorIdentifier getOpId(const Ir &ir);

private:
  bool hasLhsInplaceVariant() const final { return true; }
  std::unique_ptr<Op> getLhsInplaceVariant() const final;
  OperatorIdentifier getLhsOperatorIdentifier() const final;
};

class PowLhsInplaceOp : public ElementWiseBinaryInplaceLhsOp<PowLhsInplaceOp> {
public:
  PowLhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::PowLhsInplace,
                                      _settings) {}
};

class PowArg0GradOp : public ElementWiseBinaryArg0GradOp<PowArg0GradOp> {
public:
  PowArg0GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
};

class PowArg1GradOp : public ElementWiseBinaryArg1GradOp<PowArg1GradOp> {
public:
  PowArg1GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
};

} // namespace popart

#endif
