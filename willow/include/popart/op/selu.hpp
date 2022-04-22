// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SELU_HPP
#define GUARD_NEURALNET_SELU_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class SeluOp : public ElementWiseUnaryOp {
public:
  SeluOp(const OperatorIdentifier &opid,
         float _alpha,
         float _gamma,
         const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }
  float getGamma() const { return gamma; }

private:
  float alpha;
  float gamma;
};

class SeluInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  SeluInplaceOp(const SeluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }
  float getGamma() const { return gamma; }

private:
  float alpha;
  float gamma;
};

class SeluGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  SeluGradOp(const SeluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(OpSerialiserBase &) const override;
  float getAlpha() const { return alpha; }
  float getGamma() const { return gamma; }

private:
  float alpha;
  float gamma;
};

} // namespace popart

#endif
