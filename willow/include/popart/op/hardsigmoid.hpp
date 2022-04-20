// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HARDSIGMOID_HPP
#define GUARD_NEURALNET_HARDSIGMOID_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class HardSigmoidOp : public ElementWiseUnaryOp {
public:
  HardSigmoidOp(const OperatorIdentifier &opid,
                float _alpha,
                float _beta,
                const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }
  float getBeta() const { return beta; }

private:
  float alpha;
  float beta;
};

class HardSigmoidInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  HardSigmoidInplaceOp(const HardSigmoidOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }
  float getBeta() const { return beta; }

private:
  float alpha;
  float beta;
};

class HardSigmoidGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  HardSigmoidGradOp(const HardSigmoidOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }
  float getBeta() const { return beta; }

private:
  float alpha;
  float beta;
};

} // namespace popart

#endif
