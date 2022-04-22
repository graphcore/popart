// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_INCREMENTMOD_HPP
#define GUARD_NEURALNET_INCREMENTMOD_HPP

#include <memory>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

/**
 * Increment Modulo Op
 *
 * This Op takes one Tensor as input (as indicated in \see
 *graphcoreoperators.hpp)
 * 1. The Tensor to increment (modulo)
 * The output is the tensor  y = (x + increment) % modulus
 *
 * Attributes:
 * 1. increment - how much to increment the input tensor by (const scalar)
 * 2. modulus - the modulo operand (const scalar)
 **/
class IncrementModOp : public ElementWiseUnaryOp {
public:
  IncrementModOp(const OperatorIdentifier &opId,
                 double increment_,
                 double modulus_,
                 const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  double getIncrement() const { return increment; }
  double getModulus() const { return modulus; }

private:
  double increment;
  double modulus;
};

/**
 * Increment Modulo Op
 *
 * This Op takes one Tensor as input (as indicated in \see
 *graphcoreoperators.hpp)
 * 1. The Tensor to increment (modulo)
 * The output is the tensor  x = (x + increment) % modulus
 *
 * Attributes:
 * 1. increment - how much to increment the input tensor by (const scalar)
 * 2. modulus - the modulo operand (const scalar)
 *
 * Inplace - result is mapped back to the input Tensor.
 **/
class IncrementModInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  IncrementModInplaceOp(double increment_,
                        double modulus_,
                        const Op::Settings &settings);
  IncrementModInplaceOp(const IncrementModOp &);
  std::unique_ptr<Op> clone() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  double getIncrement() const { return increment; }
  double getModulus() const { return modulus; }

private:
  double increment;
  double modulus;
};

} // namespace popart

#endif // !GUARD_NEURALNET_INCREMENTMOD_HPP
