// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_MEAN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_MEAN_HPP_

#include <memory>
#include <vector>
#include <popart/op/variadic.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class MeanOp : public VariadicOp {
public:
  MeanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;

private:
  virtual std::unique_ptr<Op> getIthGrad(int) const final;
};

class MeanArgGradOp : public LinearVariadicGradOp {
public:
  MeanArgGradOp(const MeanOp &, InIndex inIndex);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  std::unique_ptr<Op> clone() const final;

  bool hasScale() const final { return true; }
  float getScale() const final { return 1.0f / static_cast<float>(nInputs); }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

private:
  std::vector<GradInOutMapper> gradInputInfoVec;
  int nInputs;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_MEAN_HPP_
