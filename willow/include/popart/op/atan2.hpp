// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATAN2_HPP
#define GUARD_NEURALNET_ATAN2_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

class Atan2Arg0GradOp;
class Atan2Arg1GradOp;

class Atan2Op
    : public ElementWiseNpBroadcastableBinaryWithGradOp<Atan2Arg0GradOp,
                                                        Atan2Arg1GradOp> {
public:
  Atan2Op(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;

private:
  bool hasLhsInplaceVariant() const final { return true; }
  std::unique_ptr<Op> getLhsInplaceVariant() const final;
  OperatorIdentifier getLhsOperatorIdentifier() const final;
};

class Atan2LhsInplaceOp : public ElementWiseBinaryInplaceLhsOp {
public:
  Atan2LhsInplaceOp(const Op::Settings &_settings)
      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::Atan2Inplace,
                                      _settings) {}
  std::unique_ptr<Op> clone() const final;
};

class Atan2Arg0GradOp : public ElementWiseBinaryArg0GradOp {
public:
  Atan2Arg0GradOp(const Op &, const std::vector<int64_t> &reduction_axes);
  std::unique_ptr<Op> clone() const final;
};

class Atan2Arg1GradOp : public ElementWiseBinaryArg1GradOp {
public:
  Atan2Arg1GradOp(const Op &, const std::vector<int64_t> &reduction_axes);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
