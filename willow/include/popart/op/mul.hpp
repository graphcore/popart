// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_MUL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_MUL_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

class MulArg0GradOp;
class MulArg1GradOp;
class Ir;

class MulOp : public ElementWiseNpBroadcastableBinaryWithGradOp<MulArg0GradOp,
                                                                MulArg1GradOp> {
public:
  MulOp(const OperatorIdentifier &_opid, const Op::Settings &_settings);
  std::unique_ptr<Op> clone() const final;
  static OperatorIdentifier getOpId(const Ir &ir);

  // Specialised output type inference because of popops::mul support for
  // mixed-precision inputs
  void setup() final;

private:
  bool hasLhsInplaceVariant() const final { return true; }
  bool hasRhsInplaceVariant() const final { return true; }

  std::unique_ptr<Op> getLhsInplaceVariant() const final;
  std::unique_ptr<Op> getRhsInplaceVariant() const final;

  OperatorIdentifier getLhsOperatorIdentifier() const final;
  OperatorIdentifier getRhsOperatorIdentifier() const final;
};

class MulLhsInplaceOp : public ElementWiseBinaryInplaceLhsOp {
public:
  MulLhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::MulLhsInplace,
                                      _settings) {}
  std::unique_ptr<Op> clone() const final;

  // Specialised output type inference because of popops::mulInPlace support for
  // mixed-precision inputs
  void setup() final;
};

class MulRhsInplaceOp : public ElementWiseBinaryInplaceRhsOp {
public:
  MulRhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceRhsOp(Onnx::CustomOperators::MulRhsInplace,
                                      _settings) {}
  std::unique_ptr<Op> clone() const final;

  // Specialised output type inference because of popops::mulInPlace support for
  // mixed-precision inputs
  void setup() final;
};

class MulArg0GradOp : public ElementWiseBinaryArg0GradOp {
public:
  MulArg0GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
  std::unique_ptr<Op> clone() const final;
};

class MulArg1GradOp : public ElementWiseBinaryArg1GradOp {
public:
  MulArg1GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_MUL_HPP_
