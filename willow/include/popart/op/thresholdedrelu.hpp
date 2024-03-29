// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_THRESHOLDEDRELU_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_THRESHOLDEDRELU_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class ThresholdedReluOp : public ElementWiseUnaryOp {
public:
  ThresholdedReluOp(const OperatorIdentifier &opid,
                    float _alpha,
                    const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }

private:
  float alpha;
};

class ThresholdedReluInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  ThresholdedReluInplaceOp(const ThresholdedReluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }

private:
  float alpha;
};

class ThresholdedReluGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  ThresholdedReluGradOp(const ThresholdedReluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }

private:
  float alpha;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_THRESHOLDEDRELU_HPP_
