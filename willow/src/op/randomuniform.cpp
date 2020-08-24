// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/randomuniform.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

RandomUniformOp::RandomUniformOp(const OperatorIdentifier &opid_,
                                 const std::vector<int64_t> &shape_,
                                 const OptionalDataType &dataType_,
                                 float high_,
                                 float low_,
                                 const Op::Settings &settings_)
    : RandomUniformBaseOp(opid_, dataType_, high_, low_, settings_),
      shape(shape_) {}

std::unique_ptr<Op> RandomUniformOp::clone() const {
  return std::make_unique<RandomUniformOp>(*this);
}

void RandomUniformOp::setup() { setupWithShape(shape); }

RandomUniformLikeOp::RandomUniformLikeOp(const OperatorIdentifier &opid_,
                                         const OptionalDataType &dataType_,
                                         float high_,
                                         float low_,
                                         const Op::Settings &settings_)
    : RandomUniformBaseOp(opid_, dataType_, high_, low_, settings_) {}

std::unique_ptr<Op> RandomUniformLikeOp::clone() const {
  return std::make_unique<RandomUniformLikeOp>(*this);
}

void RandomUniformLikeOp::setup() { setupLike(inInfo(getInIndex())); }

std::vector<std::unique_ptr<Op>> RandomUniformLikeOp::getGradOps() {
  throw error("RandomUniformLikeOp should be removed by pattern "
              "'RandomUniformLikeOpPattern' before call to "
              "RandomUniformLikeOp::getGradOps");
}

std::unique_ptr<RandomUniformOp>
RandomUniformLikeOp::foldInputTensor(const Op::Settings &settings) const {
  const auto &input = inTensor(getInIndex())->info;

  return std::make_unique<RandomUniformOp>(Onnx::Operators::RandomUniform_1,
                                           input.shape(),
                                           input.dataType(),
                                           getHigh(),
                                           getLow(),
                                           settings);
}

namespace {

static OpDefinition randomUniformOpDef(
    {OpDefinition::Inputs({}),
     OpDefinition::Outputs({{"output", RandomBaseOp::getSupportedDataTypes()}}),
     OpDefinition::Attributes({{"shape", {"*"}},
                               {"dtype", {"*"}},
                               {"high", {"*"}},
                               {"low", {"*"}},
                               {"seed", {"*"}}})});

static OpCreator<RandomUniformOp> randomUniformOpCreator(
    OpDefinitions({{Onnx::Operators::RandomUniform_1, randomUniformOpDef}}),
    [](const OpCreatorInfo &info) {
      const auto &attr = info.attributes;
      auto shape       = attr.getAttribute<Attributes::Ints>("shape");
      float high       = attr.getAttribute<Attributes::Float>("high", 1.0f);
      float low        = attr.getAttribute<Attributes::Float>("low", 0.0f);
      auto dataType    = RandomBaseOp::getOptionalDataType(attr, info.opid);
      RandomBaseOp::errorIfSeedIsSet(attr, info.opid);

      return std::unique_ptr<Op>(new RandomUniformOp(
          info.opid, shape, dataType, high, low, info.settings));
    },
    /*isPublic=*/true);

// RandomUniformLike: Constrain to any tensor type. If the dtype attribute is
// not provided this must be a valid output type.
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

static OpDefinition randomUniformLikeOpDef(
    {OpDefinition::Inputs({{"inputs", T}}),
     OpDefinition::Outputs({{"output", RandomBaseOp::getSupportedDataTypes()}}),
     OpDefinition::Attributes({{"dtype", {"*"}},
                               {"high", {"*"}},
                               {"low", {"*"}},
                               {"seed", {"*"}}})});

static OpCreator<RandomUniformLikeOp> randomUniformLikeOpCreator(
    OpDefinitions({{Onnx::Operators::RandomUniformLike_1,
                    randomUniformLikeOpDef}}),
    [](const OpCreatorInfo &info) {
      const auto &attr = info.attributes;
      float high       = attr.getAttribute<Attributes::Float>("high", 1.0f);
      float low        = attr.getAttribute<Attributes::Float>("low", 0.0f);
      auto dataType    = RandomBaseOp::getOptionalDataType(attr, info.opid);
      RandomBaseOp::errorIfSeedIsSet(attr, info.opid);

      return std::unique_ptr<Op>(new RandomUniformLikeOp(
          info.opid, dataType, high, low, info.settings));
    },
    /*isPublic=*/true);
} // namespace

} // namespace popart
