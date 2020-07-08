// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/op/lamb.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

LambSquareOp::LambSquareOp(const Op::Settings &settings_)
    : Op(Onnx::CustomOperators::LambSquare, settings_) {}

std::unique_ptr<Op> LambSquareOp::clone() const {
  return std::make_unique<LambSquareOp>(*this);
}

void LambSquareOp::setup() { outInfo(getOutIndex()) = {DataType::FLOAT, {}}; }

} // namespace popart
