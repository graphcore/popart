// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATAN2_HPP
#define GUARD_NEURALNET_ATAN2_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class Atan2Op : public ElementWiseBinaryOp {
public:
  Atan2Op(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;

private:
  bool hasLhsInplaceVariant() const final { return true; }

  std::unique_ptr<Op> getLhsInplaceVariant() const final;

  OperatorIdentifier getLhsOperatorIdentifier() const final;
};

class Atan2LhsInplaceOp
    : public ElementWiseBinaryInplaceLhsOp<Atan2LhsInplaceOp> {
public:
  Atan2LhsInplaceOp(const Op::Settings &settings)
      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::Atan2Inplace,
                                      settings) {}
};

} // namespace popart

#endif
