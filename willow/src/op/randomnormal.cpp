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
                               DataType dataType_,
                               float mean_,
                               float scale_,
                               const Op::Settings &settings_)
    : Op(opid_, settings_), shape(shape_), dataType(dataType_), mean(mean_),
      scale(scale_),
      seedModifier(settings_.getIr().getAndIncrementSeedModifier()) {}

std::unique_ptr<Op> RandomNormalOp::clone() const {
  return std::make_unique<RandomNormalOp>(*this);
}

void RandomNormalOp::setup() {
  outInfo(getOutIndex()) = TensorInfo(dataType, shape);
}

void RandomNormalOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("mean", mean);
  os.appendAttribute("scale", scale);
  os.appendAttribute("seedModifier", seedModifier);
}

namespace {

static OpDefinition::DataTypes supportedDataTypes = {DataType::FLOAT16,
                                                     DataType::FLOAT};

static OpDefinition
    randomNormalOpDef({OpDefinition::Inputs({}),
                       OpDefinition::Outputs({{"output", supportedDataTypes}}),
                       OpDefinition::Attributes({{"shape", {"*"}},
                                                 {"dtype", {"*"}},
                                                 {"mean", {"*"}},
                                                 {"scale", {"*"}},
                                                 {"seed", {"*"}}})});

static OpCreator<RandomNormalOp> randomNormalOpCreator(
    OpDefinitions({{Onnx::Operators::RandomNormal_1, randomNormalOpDef}}),
    [](const OpCreatorInfo &info) {
      const auto &attr = info.attributes;

      auto shape  = attr.getAttribute<Attributes::Ints>("shape");
      float mean  = attr.getAttribute<Attributes::Float>("mean", 0.0f);
      float scale = attr.getAttribute<Attributes::Float>("scale", 1.0f);

      constexpr int dtypeDefaultValue =
          ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
      auto onnxDataType =
          attr.getAttribute<Attributes::Int>("dtype", dtypeDefaultValue);

      logging::trace("RandomNormal : dtype: {}", onnxDataType);
      DataType dataType = onnxutil::getDataType(onnxDataType);
      logging::trace("RandomNormal : popart DataType {}",
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

      return std::unique_ptr<Op>(new RandomNormalOp(
          info.opid, shape, dataType, mean, scale, info.settings));
    },
    /*isPublic=*/true);

} // namespace

} // namespace popart
