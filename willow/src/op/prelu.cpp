// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/prelu.hpp>
#include <popart/opmanager.hpp>

namespace popart {

PReluOp::PReluOp(const OperatorIdentifier &opid_, const Op::Settings &settings_)
    : ElementWiseBinaryOp(opid_, settings_) {}

std::unique_ptr<Op> PReluOp::clone() const {
  return std::make_unique<PReluOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition prelu9_OpDef({OpDefinition::Inputs({
                                      {"X", T},
                                      {"slope", T},
                                  }),
                                  OpDefinition::Outputs({{"Y", T}}),
                                  OpDefinition::Attributes({})});

static OpCreator<PReluOp> prelu9_OpCreator(
    OpDefinitions({{Onnx::Operators::PRelu_9, prelu9_OpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      return std::unique_ptr<Op>(new PReluOp(info.opid, info.settings));
    },
    true);

} // namespace

} // namespace popart
