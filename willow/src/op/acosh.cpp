// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/acosh.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

AcoshOp::AcoshOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> AcoshOp::clone() const {
  return std::make_unique<AcoshOp>(*this);
}

std::vector<std::unique_ptr<Op>> AcoshOp::getGradOps() {
  throw error(
      "AcoshOp should be removed by pattern 'AcoshOpPattern' before call to "
      "AcoshOp::getGradOps");
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition acoshOpDef({OpDefinition::Inputs({
                                    {"input", T},
                                }),
                                OpDefinition::Outputs({{"output", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<AcoshOp>
    acoshOpCreator(OpDefinitions({{Onnx::Operators::Acosh_9, acoshOpDef}}));

} // namespace

} // namespace popart
