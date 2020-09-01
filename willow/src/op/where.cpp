// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <string>
#include <vector>

#include <memory>
#include <popart/op/where.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

WhereOp::WhereOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> WhereOp::clone() const {
  return std::make_unique<WhereOp>(*this);
}

void WhereOp::setup() {
  outInfo(outIndex()) =
      TensorInfo(inInfo(xInIndex()).dataType(), inShape(conditionInIndex()));
  outInfo(outIndex()) = prettyNpOut(outInfo(outIndex()), inInfo(xInIndex()));
  outInfo(outIndex()) = prettyNpOut(outInfo(outIndex()), inInfo(yInIndex()));
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
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};
static OpDefinition::DataTypes TB = {DataType::BOOL};

static OpDefinition
    whereOpDef({OpDefinition::Inputs({{"condition", TB}, {"X", T}, {"Y", T}}),
                OpDefinition::Outputs({{"output", T}}),
                OpDefinition::Attributes({})});

static OpCreator<WhereOp>
    whereOpCreator(OpDefinitions({{Onnx::Operators::Where_9, whereOpDef}}));
} // namespace

} // namespace popart
