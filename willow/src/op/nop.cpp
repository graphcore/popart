// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/identity.hpp>
#include <popart/op/nop.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

NopOp::NopOp(const OperatorIdentifier &opid_, const Op::Settings &settings_)
    : ElementWiseUnaryOp(opid_, settings_) {}

std::unique_ptr<Op> NopOp::clone() const {
  return std::make_unique<NopOp>(*this);
}

std::vector<std::unique_ptr<Op>> NopOp::getGradOps() { return {}; }

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};

static OpDefinition printTensorOpDef({OpDefinition::Inputs({{"X", T}}),
                                      OpDefinition::Outputs({{"Y", T}}),
                                      OpDefinition::Attributes({})});

static OpCreator<NopOp> nopOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::Nop_1, printTensorOpDef},
    }),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new NopOp(info.opid, info.settings));
    },
    true);
} // namespace

} // namespace popart
