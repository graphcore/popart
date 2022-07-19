// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_BITWISE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_BITWISE_HPP_

#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {
struct OperatorIdentifier;

// This class implements bitwise not operator.
class BitwiseNotOp : public ElementWiseUnaryOp {
public:
  BitwiseNotOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

// This class implements all binary bitwise operators.
class BitwiseBinaryOp : public ElementWiseBinaryOp {
public:
  BitwiseBinaryOp(const OperatorIdentifier &_opid,
                  const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_BITWISE_HPP_
