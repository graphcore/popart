// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SUM_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SUM_HPP_

#include <memory>
#include <vector>
#include <popart/op/variadic.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class SumOp : public VariadicOp {
public:
  SumOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;

  bool canShard() const override { return true; }

private:
  virtual std::unique_ptr<Op> getIthGrad(int) const final;
};

class SumArgGradOp : public LinearVariadicGradOp {
public:
  SumArgGradOp(const SumOp &, InIndex inIndex);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  std::unique_ptr<Op> clone() const final;

  bool canBeReplacedByIdentity() const override;

  bool canShard() const override { return true; }

private:
  std::vector<GradInOutMapper> gradInputInfoVec;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SUM_HPP_
