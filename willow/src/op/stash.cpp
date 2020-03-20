// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/stash.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

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
