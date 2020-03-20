// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <vector>
#include <popart/op/less.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

LessOp::LessOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> LessOp::clone() const {
  return std::make_unique<LessOp>(*this);
}

std::vector<std::unique_ptr<Op>> LessOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to LessOp");
}

namespace {

static OpDefinition::DataTypes T  = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT,
                                    DataType::FLOAT16};
static OpDefinition::DataTypes T1 = {DataType::BOOL};

static OpDefinition lessOpDef({OpDefinition::Inputs({{"A", T}, {"B", T}}),
                               OpDefinition::Outputs({{"B", T1}}),
                               OpDefinition::Attributes({})});

static OpCreator<LessOp>
    LessOpCreator(OpDefinitions({{Onnx::Operators::Less_7, lessOpDef},
                                 {Onnx::Operators::Less_9, lessOpDef}}));
} // namespace

} // namespace popart
