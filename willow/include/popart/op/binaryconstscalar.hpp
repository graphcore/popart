// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BINARY_CONST_SCALAR_HPP
#define GUARD_NEURALNET_BINARY_CONST_SCALAR_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

/**
 * A unary Op, which performs a binary operation (Mul, Div, etc) between its
 * single input tensor and a scalar, whose value is stored as an Op attribute.
 * The input index (0 or 1) of the tensor and scalar are controlled by the
 * scalarInIndex attribute.
 *
 * Some examples. Let T be the input tensor of this Op.
 *
 * [value = 2, opType = "Div", scalarInIndex = 1]:
 *    T / 2.0
 *
 * [value = 4, opType = "Pow", scalarInIndex = 0]:
 *    2.0 ** T
 *
 * [value = 0.2, opType = "Add", scalarInIndex = 0]:
 *  0.2 + T
 *
 * [value = 100, opType = "Sub", scalarInIndex = 1]:
 *   T - 100.
 */
class BinaryConstScalarOp : public ElementWiseUnaryOp {
public:
  enum class Type {
    Add = 0,
    Sub,
    Mul,
    Div,
    Pow,
    N /*Not a type, counts number of Types */
  };

  BinaryConstScalarOp(const OperatorIdentifier &x,
                      float value,
                      Type t,
                      int64_t index,
                      const Op::Settings &settings)
      : ElementWiseUnaryOp(x, settings), value_(value), opType_(t),
        scalarInIndex_(index) {}

  std::unique_ptr<Op> clone() const override;

  // The Op will be eliminated by a Pattern before auto-grad is run, so we don't
  // need to implement this method.
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  float value() const { return value_; }
  Type opType() const { return opType_; }
  int64_t scalarInIndex() const { return scalarInIndex_; }

private:
  float value_;
  Type opType_;
  int64_t scalarInIndex_;
};

} // namespace popart

#endif
