// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SHRINK_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SHRINK_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class ShrinkOp : public ElementWiseUnaryOp {
public:
  ShrinkOp(const OperatorIdentifier &opid,
           float lambd,
           float bias,
           const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float lambd() const { return lambd_; }
  float bias() const { return bias_; }

private:
  float lambd_;
  float bias_;
};

class ShrinkInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  ShrinkInplaceOp(const ShrinkOp &);
  std::unique_ptr<Op> clone() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float lambd() const { return lambd_; }
  float bias() const { return bias_; }

private:
  float lambd_;
  float bias_;
};

class ShrinkGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  ShrinkGradOp(const ShrinkOp &);
  std::unique_ptr<Op> clone() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float lambd() const { return lambd_; }
  float bias() const { return bias_; }

private:
  float lambd_;
  float bias_;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SHRINK_HPP_
