// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SHAPEORLIKE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SHAPEORLIKE_HPP_

#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;
class TensorInfo;

// Shared base class for ops with a shape or "-Like" version the latter being
// ops which are ops which output a value in the same shape and datatype as an
// input but but otherwise unrelated to it
class ShapeOrLikeOp : public Op {
public:
  ShapeOrLikeOp(const OperatorIdentifier &opid_,
                const OptionalDataType &dataType_,
                const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override = 0;

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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SHAPEORLIKE_HPP_
