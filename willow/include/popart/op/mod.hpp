// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MOD_HPP
#define GUARD_NEURALNET_MOD_HPP

#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

// Implements ONNX Mod operator.
class ModOp : public ElementWiseBinaryOp {
public:
  ModOp(const OperatorIdentifier &opId, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ModArg0GradOp : public ElementWiseBinaryArg0GradOp<ModArg0GradOp> {
public:
  ModArg0GradOp(const ModOp &op, const std::vector<int64_t> &reductionAxes);
};

} // namespace popart

#endif // !GUARD_NEURALNET_MOD_HPP
