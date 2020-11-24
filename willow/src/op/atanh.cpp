// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/atanh.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

AtanhOp::AtanhOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> AtanhOp::clone() const {
  return std::make_unique<AtanhOp>(*this);
}

std::vector<std::unique_ptr<Op>> AtanhOp::getGradOps() {
  throw error(
      "AtanhOp should be removed by pattern 'AtanhOpPattern' before call to "
      "AtanhOp::getGradOps");
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition atanhOpDef({OpDefinition::Inputs({
                                    {"input", T},
                                }),
                                OpDefinition::Outputs({{"output", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<AtanhOp>
    atanhOpCreator(OpDefinitions({{Onnx::Operators::Atanh_9, atanhOpDef}}));

} // namespace

} // namespace popart
