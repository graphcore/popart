// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/acos.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

AcosOp::AcosOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> AcosOp::clone() const {
  return std::make_unique<AcosOp>(*this);
}

std::vector<std::unique_ptr<Op>> AcosOp::getGradOps() {
  throw error(
      "AcosOp should be removed by pattern 'AcosOpPattern' before call to "
      "AcosOp::getGradOps");
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition acosOpDef({OpDefinition::Inputs({
                                   {"input", T},
                               }),
                               OpDefinition::Outputs({{"output", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<AcosOp>
    acosOpCreator(OpDefinitions({{Onnx::Operators::Acos_7, acosOpDef}}));

} // namespace

} // namespace popart
