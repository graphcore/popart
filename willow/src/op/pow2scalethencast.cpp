// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <vector>
#include <popart/op/pow2scalethencast.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

//
// ***** Cast to float8 ***** //
//
Pow2ScaleThenCastOp::Pow2ScaleThenCastOp(const OperatorIdentifier &_opid,
                                         const DataType _to,
                                         const Op::Settings &settings_)
    : Op(_opid, settings_), to(_to) {}

std::unique_ptr<Op> Pow2ScaleThenCastOp::clone() const {
  return std::make_unique<Pow2ScaleThenCastOp>(*this);
}

std::vector<std::unique_ptr<Op>> Pow2ScaleThenCastOp::getGradOps() {
  throw error(
      "Floating point 8 is not supported for training. This is for op {}.",
      this->debugName());
}

void Pow2ScaleThenCastOp::setup() {
  auto info = inInfo(getInIndex());

  // Change data type
  info.set(to);
  outInfo(getOutIndex()) = info;
}

bool Pow2ScaleThenCastOp::canBeReplacedByIdentity() const { return false; }
namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT8_143,
                                     DataType::FLOAT8_152};
static OpDefinition::DataTypes T2 = {DataType::FLOAT16, DataType::FLOAT};

} // namespace

static OpDefinition pow2ScaleThenCastOpDef(
    {OpDefinition::Inputs({{"input", T2}, {"metadata", {DataType::INT8}}}),
     OpDefinition::Outputs({{"output", T1}}),
     OpDefinition::Attributes({{"to", {"FLOAT8_143|FLOAT8_152"}}})});

static OpCreator<Pow2ScaleThenCastOp> pow2ScaleThenCastOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::Pow2ScaleThenCast, pow2ScaleThenCastOpDef},
    }),
    [](const OpCreatorInfo &info) {
      auto to    = info.attributes.getAttribute<Attributes::String>("to");
      auto dt_to = dataTypeFromString(to);

      return std::make_unique<Pow2ScaleThenCastOp>(
          info.opid, dt_to, info.settings);
    },
    true);

} // namespace popart
