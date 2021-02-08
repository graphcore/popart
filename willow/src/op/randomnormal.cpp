// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/randomnormal.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

RandomNormalOp::RandomNormalOp(const OperatorIdentifier &opid_,
                               const std::vector<int64_t> &shape_,
                               const OptionalDataType &dataType_,
                               float mean_,
                               float scale_,
                               const Op::Settings &settings_,
                               RandomSeedPlaceholder placeholder_)
    : RandomNormalBaseOp(opid_,
                         dataType_,
                         mean_,
                         scale_,
                         settings_,
                         placeholder_),
      shape(shape_) {}

std::unique_ptr<Op> RandomNormalOp::clone() const {
  return std::make_unique<RandomNormalOp>(*this);
}

void RandomNormalOp::setup() { setupWithShape(shape); }

RandomNormalLikeOp::RandomNormalLikeOp(const OperatorIdentifier &opid_,
                                       const OptionalDataType &dataType_,
                                       float mean_,
                                       float scale_,
                                       const Op::Settings &settings_,
                                       RandomSeedPlaceholder placeholder_)
    : RandomNormalBaseOp(opid_,
                         dataType_,
                         mean_,
                         scale_,
                         settings_,
                         placeholder_) {}

std::unique_ptr<Op> RandomNormalLikeOp::clone() const {
  return std::make_unique<RandomNormalLikeOp>(*this);
}

void RandomNormalLikeOp::setup() { setupLike(inInfo(getInIndex())); }

std::vector<std::unique_ptr<Op>> RandomNormalLikeOp::getGradOps() {
  throw error("RandomNormalLikeOp should be removed by pattern "
              "'RandomNormalLikeOpPattern' before call to "
              "RandomNormalLikeOp::getGradOps");
}

std::unique_ptr<RandomNormalOp>
RandomNormalLikeOp::foldInputTensor(const Op::Settings &settings) const {
  const auto &input = inTensor(getInIndex())->info;

  return std::make_unique<RandomNormalOp>(Onnx::Operators::RandomNormal_1,
                                          input.shape(),
                                          input.dataType(),
                                          getMean(),
                                          getScale(),
                                          settings);
}

namespace {

static OpDefinition randomNormalOpDef(
    {OpDefinition::Inputs({}),
     OpDefinition::Outputs({{"output", RandomBaseOp::supportedDataTypes()}}),
     OpDefinition::Attributes({{"shape", {"*"}},
                               {"dtype", {"*"}},
                               {"mean", {"*"}},
                               {"scale", {"*"}},
                               {"seed", {"*"}}})});

static OpCreator<RandomNormalOp> randomNormalOpCreator(
    OpDefinitions({{Onnx::Operators::RandomNormal_1, randomNormalOpDef}}),
    [](const OpCreatorInfo &info) {
      const auto &attr = info.attributes;
      auto shape       = attr.getAttribute<Attributes::Ints>("shape");
      float mean       = attr.getAttribute<Attributes::Float>("mean", 0.0f);
      float scale      = attr.getAttribute<Attributes::Float>("scale", 1.0f);
      auto dataType    = RandomBaseOp::getOptionalDataType(attr, info.opid);
      RandomBaseOp::errorIfSeedIsSet(attr, info.opid);

      return std::unique_ptr<Op>(new RandomNormalOp(
          info.opid, shape, dataType, mean, scale, info.settings));
    },
    /*isPublic=*/true);

// RandomNormalLike: Constrain to any tensor type. If the dtype attribute is not
// provided this must be a valid output type.
static OpDefinition::DataTypes T = {
    DataType::UINT8,
    DataType::INT8,
    DataType::UINT16,
    DataType::INT16,
    DataType::INT32,
    DataType::INT64,
    DataType::UINT32,
    DataType::UINT64,
    DataType::BOOL,
    DataType::FLOAT,
    DataType::FLOAT16,
    DataType::BFLOAT16,
    DataType::DOUBLE,
    DataType::COMPLEX64,
    DataType::COMPLEX128,
    DataType::STRING,
};

static OpDefinition randomNormalLikeOpDef(
    {OpDefinition::Inputs({{"inputs",
                            ShapeOrLikeOp::likeSupportedInputTypes()}}),
     OpDefinition::Outputs({{"output", RandomBaseOp::supportedDataTypes()}}),
     OpDefinition::Attributes({{"dtype", {"*"}},
                               {"mean", {"*"}},
                               {"scale", {"*"}},
                               {"seed", {"*"}}})});

static OpCreator<RandomNormalLikeOp> randomNormalLikeOpCreator(
    OpDefinitions({{Onnx::Operators::RandomNormalLike_1,
                    randomNormalLikeOpDef}}),
    [](const OpCreatorInfo &info) {
      const auto &attr = info.attributes;
      float mean       = attr.getAttribute<Attributes::Float>("mean", 0.0f);
      float scale      = attr.getAttribute<Attributes::Float>("scale", 1.0f);
      auto dataType    = RandomBaseOp::getOptionalDataType(attr, info.opid);
      RandomBaseOp::errorIfSeedIsSet(attr, info.opid);

      return std::unique_ptr<Op>(new RandomNormalLikeOp(
          info.opid, dataType, mean, scale, info.settings));
    },
    /*isPublic=*/true);

} // namespace
} // namespace popart