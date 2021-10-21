// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SHAPEORLIKEOP_HPP
#define GUARD_NEURALNET_SHAPEORLIKEOP_HPP

#include <vector>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>

namespace popart {

// Shared base class for ops with a shape or "-Like" version the latter being
// ops which are ops which output a value in the same shape and datatype as an
// input but but otherwise unrelated to it
class ShapeOrLikeOp : public Op {
public:
  ShapeOrLikeOp(const OperatorIdentifier &opid_,
                const OptionalDataType &dataType_,
                const Op::Settings &settings_);

  static OptionalDataType getOptionalDataType(const Attributes &attr,
                                              OperatorIdentifier opid);

  static OutIndex getOutIndex() { return 0; }

  static const OpDefinition::DataTypes &likeSupportedInputTypes();

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  void validateDataType(DataType dataType, OperatorIdentifier opid);

  const OptionalDataType &getDataType() const { return dataType; }

  virtual std::vector<DataType> getSupportedDataTypes() const = 0;

protected:
  void setupLike(const popart::TensorInfo &info);

  void setupWithShape(const Shape &shape);

  OptionalDataType dataType;
};

} // namespace popart

#endif
