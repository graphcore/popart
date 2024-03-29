// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/stash.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensornames.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

StashOp::StashOp(const OperatorIdentifier &_opid,
                 int64_t stashSize_,
                 const Op::Settings &settings_)
    : Op(_opid, settings_), stashSize(stashSize_) {}

std::unique_ptr<Op> StashOp::clone() const {
  return std::make_unique<StashOp>(*this);
}

void StashOp::setup() {
  Shape output_shape = inShape(getInIndex());
  output_shape.insert(output_shape.begin(), getStashSize());

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), output_shape};
}

void StashOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("stashSize", stashSize);
}

int64_t StashOp::getStashSize() { return stashSize; }

TensorId StashOp::getStashedTensorId() const {
  return reservedStashedPrefix() + inId(getInIndex());
}

} // namespace popart
