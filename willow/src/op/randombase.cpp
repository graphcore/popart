// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <memory>
#include <vector>
#include <popart/op/randombase.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/op/shapeorlike.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

std::vector<DataType> RandomBaseOp::supportedDataTypes() {
  return {DataType::FLOAT16, DataType::FLOAT};
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
    : ShapeOrLikeOp(opid_, dataType_, settings_) {}

std::unique_ptr<Op> RandomBaseOp::clone() const {
  return std::make_unique<RandomBaseOp>(*this);
}

RandomNormalBaseOp::RandomNormalBaseOp(const OperatorIdentifier &opid_,
                                       const OptionalDataType &dataType_,
                                       float mean_,
                                       float scale_,
                                       const Op::Settings &settings_)
    : RandomBaseOp(opid_, dataType_, settings_), mean(mean_), scale(scale_) {}

std::unique_ptr<Op> RandomNormalBaseOp::clone() const {
  return std::make_unique<RandomNormalBaseOp>(*this);
}

void RandomNormalBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("mean", mean);
  os.appendAttribute("scale", scale);
}

RandomUniformBaseOp::RandomUniformBaseOp(const OperatorIdentifier &opid_,
                                         const OptionalDataType &dataType_,
                                         float high_,
                                         float low_,
                                         const Op::Settings &settings_)
    : RandomBaseOp(opid_, dataType_, settings_), high(high_), low(low_) {}

std::unique_ptr<Op> RandomUniformBaseOp::clone() const {
  return std::make_unique<RandomUniformBaseOp>(*this);
}

void RandomUniformBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("high", high);
  os.appendAttribute("low", low);
}

} // namespace popart
