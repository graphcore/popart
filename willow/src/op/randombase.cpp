// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/randombase.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

std::vector<DataType> RandomBaseOp::getSupportedDataTypes() {
  return {DataType::FLOAT16, DataType::FLOAT};
}

void RandomBaseOp::validateDataType(DataType dataType,
                                    OperatorIdentifier opid) {
  auto supportedDataTypes = RandomBaseOp::getSupportedDataTypes();
  bool isSupported        = std::count(supportedDataTypes.begin(),
                                supportedDataTypes.end(),
                                dataType) == 1;

  if (!isSupported) {
    throw error("{} : Unsupported data type requested: {}",
                opid,
                getDataTypeInfoMap().at(dataType).name());
  }
}

OptionalDataType RandomBaseOp::getOptionalDataType(const Attributes &attr,
                                                   OperatorIdentifier opid) {
  OptionalDataType dataType;

  if (attr.hasAttribute("dtype")) {
    auto onnxDataType = attr.getAttribute<Attributes::Int>("dtype");
    dataType          = onnxutil::getDataType(onnxDataType);
    RandomBaseOp::validateDataType(*dataType, opid);
  }

  return dataType;
}

void RandomBaseOp::errorIfSeedIsSet(const Attributes &attr,
                                    OperatorIdentifier opid) {
  if (attr.hasAttribute("seed")) {
    throw error("{}: Optional seed attribute is not supported. "
                "Use session::setRandomSeed instead.",
                opid);
  }
}

RandomBaseOp::RandomBaseOp(const OperatorIdentifier &opid_,
                           const OptionalDataType &dataType_,
                           const Op::Settings &settings_)
    : Op(opid_, settings_), dataType(dataType_),
      seedModifier(settings_.getIr().getAndIncrementSeedModifier()) {}

void RandomBaseOp::setupWithShape(const std::vector<int64_t> &shape) {
  // Default to float dataType if not specified.
  if (!dataType) {
    dataType = popart::DataType::FLOAT;
  }

  outInfo(getOutIndex()) = TensorInfo(*dataType, shape);
}

void RandomBaseOp::setupLike(const popart::TensorInfo &info) {
  // Default to using the input tensor dataType if not specified
  if (!dataType) {
    validateDataType(info.dataType(), opid);
    dataType = info.dataType();
  }

  outInfo(getOutIndex()) = TensorInfo(*dataType, info.shape());
}

RandomNormalBaseOp::RandomNormalBaseOp(const OperatorIdentifier &opid_,
                                       const OptionalDataType &dataType_,
                                       float mean_,
                                       float scale_,
                                       const Op::Settings &settings_)
    : RandomBaseOp(opid_, dataType_, settings_), mean(mean_), scale(scale_) {}

void RandomNormalBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("mean", mean);
  os.appendAttribute("scale", scale);
  os.appendAttribute("seedModifier", getSeedModifier());
}

RandomUniformBaseOp::RandomUniformBaseOp(const OperatorIdentifier &opid_,
                                         const OptionalDataType &dataType_,
                                         float high_,
                                         float low_,
                                         const Op::Settings &settings_)
    : RandomBaseOp(opid_, dataType_, settings_), high(high_), low(low_) {}

void RandomUniformBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("high", high);
  os.appendAttribute("low", low);
  os.appendAttribute("seedModifier", getSeedModifier());
}

} // namespace popart