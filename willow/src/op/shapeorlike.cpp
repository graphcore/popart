// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/onnxutil.hpp>
#include <popart/op/shapeorlike.hpp>

namespace popart {

ShapeOrLikeOp::ShapeOrLikeOp(const OperatorIdentifier &opid_,
                             const OptionalDataType &dataType_,
                             const Op::Settings &settings_)
    : Op(opid_, settings_), dataType(dataType_) {}

OptionalDataType ShapeOrLikeOp::getOptionalDataType(const Attributes &attr,
                                                    OperatorIdentifier opid) {
  OptionalDataType dataType;

  if (attr.hasAttribute("dtype")) {
    auto onnxDataType = attr.getAttribute<Attributes::Int>("dtype");
    dataType          = onnxutil::getDataType(onnxDataType);
  }

  return dataType;
}

const OpDefinition::DataTypes &ShapeOrLikeOp::likeSupportedInputTypes() {
  static OpDefinition::DataTypes types{
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

  return types;
}

void ShapeOrLikeOp::validateDataType(DataType dataType,
                                     OperatorIdentifier opid) {
  auto supportedDataTypes = getSupportedDataTypes();
  bool isSupported        = std::count(supportedDataTypes.begin(),
                                supportedDataTypes.end(),
                                dataType) == 1;

  if (!isSupported) {
    throw error("{} : Unsupported data type requested: {}",
                opid,
                getDataTypeInfoMap().at(dataType).name());
  }
}

void ShapeOrLikeOp::setupWithShape(const std::vector<int64_t> &shape) {
  // Default to float dataType if not specified.
  if (!dataType) {
    dataType = popart::DataType::FLOAT;
  }

  outInfo(getOutIndex()) = TensorInfo(*dataType, shape);
}

void ShapeOrLikeOp::setupLike(const popart::TensorInfo &info) {
  // Default to using the input tensor dataType if not specified
  if (!getDataType()) {
    validateDataType(info.dataType(), opid);
    dataType = info.dataType();
  }

  outInfo(getOutIndex()) = TensorInfo(*dataType, info.shape());
}

} // namespace popart
