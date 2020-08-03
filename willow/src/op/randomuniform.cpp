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
                                 DataType dataType_,
                                 float high_,
                                 float low_,
                                 const Op::Settings &settings_)
    : Op(opid_, settings_), shape(shape_), dataType(dataType_), high(high_),
      low(low_), seedModifier(settings_.getIr().getAndIncrementSeedModifier()) {
}

std::unique_ptr<Op> RandomUniformOp::clone() const {
  return std::make_unique<RandomUniformOp>(*this);
}

void RandomUniformOp::setup() {
  outInfo(getOutIndex()) = TensorInfo(dataType, shape);
}

void RandomUniformOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("low", low);
  os.appendAttribute("high", high);
  os.appendAttribute("seedModifier", seedModifier);
}

namespace {

static OpDefinition::DataTypes supportedDataTypes = {DataType::FLOAT16,
                                                     DataType::FLOAT};

static OpDefinition
    randomUniformOpDef({OpDefinition::Inputs({}),
                        OpDefinition::Outputs({{"output", supportedDataTypes}}),
                        OpDefinition::Attributes({{"shape", {"*"}},
                                                  {"dtype", {"*"}},
                                                  {"high", {"*"}},
                                                  {"low", {"*"}},
                                                  {"seed", {"*"}}})});

static OpCreator<RandomUniformOp> randomUniformOpCreator(
    OpDefinitions({{Onnx::Operators::RandomUniform_1, randomUniformOpDef}}),
    [](const OpCreatorInfo &info) {
      const auto &attr = info.attributes;

      auto shape = attr.getAttribute<Attributes::Ints>("shape");
      float high = attr.getAttribute<Attributes::Float>("high", 1.0f);
      float low  = attr.getAttribute<Attributes::Float>("low", 0.0f);

      constexpr int dtypeDefaultValue =
          ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
      auto onnxDataType =
          attr.getAttribute<Attributes::Int>("dtype", dtypeDefaultValue);

      logging::trace("RandomUniform : dtype: {}", onnxDataType);
      DataType dataType = onnxutil::getDataType(onnxDataType);
      logging::trace("RandomUniform : popart DataType {}",
                     getDataTypeInfoMap().at(dataType).name());

      bool isSupported = std::count(supportedDataTypes.begin(),
                                    supportedDataTypes.end(),
                                    dataType) == 1;

      if (!isSupported) {
        throw error("{}: Unsupported data type requested: {}",
                    info.opid,
                    getDataTypeInfoMap().at(dataType).name());
      }

      if (attr.hasAttribute("seed")) {
        throw error("{}: Optional seed attribute is not supported. "
                    "Use session::setRandomSeed instead.",
                    info.opid);
      }

      return std::unique_ptr<Op>(new RandomUniformOp(
          info.opid, shape, dataType, high, low, info.settings));
    },
    /*isPublic=*/true);

} // namespace

} // namespace popart
