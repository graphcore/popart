// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/asinh.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

AsinhOp::AsinhOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> AsinhOp::clone() const {
  return std::make_unique<AsinhOp>(*this);
}

std::vector<std::unique_ptr<Op>> AsinhOp::getGradOps() {
  throw error(
      "AsinhOp should be removed by pattern 'AsinhOpPattern' before call to "
      "AsinhOp::getGradOps");
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition asinhOpDef({OpDefinition::Inputs({
                                    {"input", T},
                                }),
                                OpDefinition::Outputs({{"output", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<AsinhOp>
    asinhOpCreator(OpDefinitions({{Onnx::Operators::Asinh_9, asinhOpDef}}));

} // namespace

} // namespace popart
