// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ELU_HPP
#define GUARD_NEURALNET_ELU_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class EluOp : public ElementWiseUnaryOp {
public:
  EluOp(const OperatorIdentifier &opid,
        float alpha,
        const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  void appendAttributes(OpSerialiserBase &) const final;
  float alpha() const { return alpha_; }

private:
  float alpha_;
};

class EluInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  EluInplaceOp(const EluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(OpSerialiserBase &) const final;
  float alpha() const { return alpha_; }

private:
  float alpha_;
};

class EluGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  EluGradOp(const EluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(OpSerialiserBase &) const final;
  float alpha() const { return alpha_; }

private:
  float alpha_;
};

} // namespace popart

#endif
