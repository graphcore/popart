// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/ir.hpp>
#include <popart/op/randombase.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

uint64_t RandomSeedPlaceholder::placeholderCounter = 0ull;

RandomSeedPlaceholder::RandomSeedPlaceholder()
    : placeholder{std::make_shared<uint64_t>(placeholderCounter++)} {}

bool operator==(const RandomSeedPlaceholder &p0,
                const RandomSeedPlaceholder &p1) {
  return p0.placeholder == p1.placeholder;
}

bool operator!=(const RandomSeedPlaceholder &p0,
                const RandomSeedPlaceholder &p1) {
  return !(p0.placeholder == p1.placeholder);
}

bool operator<(const RandomSeedPlaceholder &p0,
               const RandomSeedPlaceholder &p1) {
  return *p0.placeholder < *p1.placeholder;
}

std::vector<DataType> RandomBaseOp::supportedDataTypes() {
  return {DataType::FLOAT16, DataType::FLOAT};
}

void RandomBaseOp::useDistinctRandomSeed() {
  setRandomSeedPlaceholder(RandomSeedPlaceholder());
}

const RandomSeedPlaceholder &RandomBaseOp::getRandomSeedPlaceholder() const {
  return placeholder;
}

void RandomBaseOp::setRandomSeedPlaceholder(
    const RandomSeedPlaceholder &placeholder_) {
  placeholder = placeholder_;
}

void RandomBaseOp::adoptRandomSeedPlaceholder(const RandomBaseOp &op) {
  placeholder = op.getRandomSeedPlaceholder();
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
                           const Op::Settings &settings_,
                           RandomSeedPlaceholder placeholder_)
    : ShapeOrLikeOp(opid_, dataType_, settings_), placeholder{placeholder_} {}

RandomNormalBaseOp::RandomNormalBaseOp(const OperatorIdentifier &opid_,
                                       const OptionalDataType &dataType_,
                                       float mean_,
                                       float scale_,
                                       const Op::Settings &settings_,
                                       RandomSeedPlaceholder placeholder_)
    : RandomBaseOp(opid_, dataType_, settings_, placeholder_), mean(mean_),
      scale(scale_) {}

void RandomNormalBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("mean", mean);
  os.appendAttribute("scale", scale);
}

RandomUniformBaseOp::RandomUniformBaseOp(const OperatorIdentifier &opid_,
                                         const OptionalDataType &dataType_,
                                         float high_,
                                         float low_,
                                         const Op::Settings &settings_,
                                         RandomSeedPlaceholder placeholder_)
    : RandomBaseOp(opid_, dataType_, settings_, placeholder_), high(high_),
      low(low_) {}

void RandomUniformBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("high", high);
  os.appendAttribute("low", low);
}

} // namespace popart
