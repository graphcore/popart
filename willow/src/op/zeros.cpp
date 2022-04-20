// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/zeros.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/shapeorlike.hpp"
#include "popart/tensor.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

ZerosBaseOp::ZerosBaseOp(const OperatorIdentifier &opid_,
                         const OptionalDataType &dataType_,
                         const Op::Settings &settings_)
    : ShapeOrLikeOp(opid_, dataType_, settings_) {}

std::unique_ptr<Op> ZerosBaseOp::clone() const {
  return std::make_unique<ZerosBaseOp>(*this);
}

std::vector<DataType> ZerosBaseOp::supportedDataTypes() {
  return {DataType::FLOAT16, DataType::FLOAT, DataType::INT32};
}

ZerosOp::ZerosOp(const OperatorIdentifier &opid_,
                 const Shape &shape_,
                 const OptionalDataType &dataType_,
                 const Op::Settings &settings_)
    : ZerosBaseOp(opid_, dataType_, settings_), shape(shape_) {}

std::unique_ptr<Op> ZerosOp::clone() const {
  return std::make_unique<ZerosOp>(*this);
}

void ZerosOp::setup() { setupWithShape(shape); }

ZerosLikeOp::ZerosLikeOp(const OperatorIdentifier &opid_,
                         const Op::Settings &settings_)
    : ZerosBaseOp(opid_, OptionalDataType(), settings_) {}

void ZerosLikeOp::setup() { setupLike(inInfo(getInIndex())); }

std::unique_ptr<Op> ZerosLikeOp::clone() const {
  return std::make_unique<ZerosLikeOp>(*this);
}

std::unique_ptr<ZerosOp>
ZerosLikeOp::foldInputTensor(const Op::Settings &settings) const {
  const auto &input = inTensor(getInIndex())->info;

  return std::make_unique<ZerosOp>(Onnx::CustomOperators::Zeros_1,
                                   input.shape(),
                                   input.dataType(),
                                   settings);
}

UnaryZeroGradOp::UnaryZeroGradOp(const OperatorIdentifier &opid_,
                                 const Op::Settings &settings_)
    : ZerosLikeOp(opid_, settings_) {}

std::unique_ptr<Op> UnaryZeroGradOp::clone() const {
  return std::make_unique<UnaryZeroGradOp>(*this);
}

std::vector<std::unique_ptr<Op>>
UnaryZeroGradOp::getGradOpVector(const Op::Settings &settings) {
  // Zeros everywhere
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<UnaryZeroGradOp>(
      Onnx::CustomGradOperators::UnaryZeroGradOp, settings));
  return upops;
}

namespace {

static OpDefinition ZerosOpDef(
    {OpDefinition::Inputs({}),
     OpDefinition::Outputs({{"output", ZerosBaseOp::supportedDataTypes()}}),
     OpDefinition::Attributes({{"shape", {"*"}}, {"dtype", {"*"}}})});

static OpCreator<ZerosOp> randomUniformOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Zeros_1, ZerosOpDef}}),
    [](const OpCreatorInfo &info) {
      const auto &attr = info.attributes;
      auto shape       = attr.getAttribute<Attributes::Ints>("shape");
      auto dataType    = ShapeOrLikeOp::getOptionalDataType(attr, info.opid);

      return std::unique_ptr<Op>(
          new ZerosOp(info.opid, shape, dataType, info.settings));
    },
    /*isPublic=*/true);

static OpDefinition zerosLikeOpDef(
    {OpDefinition::Inputs({{"inputs",
                            ShapeOrLikeOp::likeSupportedInputTypes()}}),
     OpDefinition::Outputs({{"output", ZerosBaseOp::supportedDataTypes()}}),
     OpDefinition::Attributes({{"dtype", {"*"}}})});

static OpCreator<ZerosLikeOp> zerosLikeOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ZerosLike_1, zerosLikeOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new ZerosLikeOp(info.opid, info.settings));
    },
    /*isPublic=*/true);

static OpCreator<UnaryZeroGradOp> unaryZeroGradOpCreation(
    OpDefinitions({{Onnx::CustomGradOperators::UnaryZeroGradOp,
                    zerosLikeOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new UnaryZeroGradOp(info.opid, info.settings));
    },
    /*isPublic=*/true);

} // namespace
} // namespace popart
