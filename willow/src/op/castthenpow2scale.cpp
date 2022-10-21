// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/castthenpow2scale.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/tensor.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

//
// ***** Cast from float8 ***** //
//
CastThenPow2ScaleOp::CastThenPow2ScaleOp(const OperatorIdentifier &_opid,
                                         const DataType _to,
                                         const Op::Settings &settings_)
    : Op(_opid, settings_), to(_to) {}

std::unique_ptr<Op> CastThenPow2ScaleOp::clone() const {
  return std::make_unique<CastThenPow2ScaleOp>(*this);
}

std::vector<std::unique_ptr<Op>> CastThenPow2ScaleOp::getGradOps() {
  throw error(
      "Floating point 8 is not supported for training. This is for op {}.",
      this->debugName());
}

void CastThenPow2ScaleOp::setup() {
  if (!(inTensor(getInIndex())->info.dataType() == DataType::FLOAT8_143 ||
        inTensor(getInIndex())->info.dataType() == DataType::FLOAT8_152)) {

    throw error("Casting from FLOAT8 is only supported for FLOAT8_143 and "
                "FLOAT8_152 input types. This is for op {}.",
                this->debugName());
  }
  auto info = inInfo(getInIndex());
  // Change data type
  info.set(to);
  outInfo(getOutIndex()) = info;
}

bool CastThenPow2ScaleOp::canBeReplacedByIdentity() const { return false; }

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT8_143,
                                     DataType::FLOAT8_152};
static OpDefinition::DataTypes T2 = {DataType::FLOAT16};

} // namespace

static OpDefinition castThenPow2ScaleOpDef(
    {OpDefinition::Inputs({{"input", T1}, {"metadata", {DataType::INT8}}}),
     OpDefinition::Outputs({{"output", T2}}),
     OpDefinition::Attributes({{"to", {"FLOAT16"}}})});

static OpCreator<CastThenPow2ScaleOp> castThenPow2ScaleOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::CastThenPow2Scale, castThenPow2ScaleOpDef},
    }),
    [](const OpCreatorInfo &info) {
      int64_t i64_to;
      info.attributes.set(i64_to, "to");
      DataType dt_to = DataType::FLOAT16;

      return std::make_unique<CastThenPow2ScaleOp>(
          info.opid, dt_to, info.settings);
    },
    true);

} // namespace popart
