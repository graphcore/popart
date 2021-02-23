// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/ir.hpp>
#include <popart/op/randombase.hpp>
#include <popart/opserialiser.hpp>

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
    : ShapeOrLikeOp(opid_, dataType_, settings_),
      seedModifier(settings_.getIr().getAndIncrementSeedModifier()) {}

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
