// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/op/basesort.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

BaseSortOp::BaseSortOp(const OperatorIdentifier &_opid,
                       int64_t axis_,
                       const Op::Settings &settings_)
    : Op(_opid, settings_), axis(axis_) {}

std::unique_ptr<Op> BaseSortOp::clone() const {
  return std::make_unique<BaseSortOp>(*this);
}

void BaseSortOp::validateAxis() const {
  auto shape = inShape(getInIndex());

  if (shape.size() == 0) {
    throw error(
        "BaseSortOp input rank must be greater than 0, invalid BaseSortOp {}.",
        str());
  }

  if (shape.size() <= axis) {
    throw error("Cannot compute BaseSortOp on axis {} when input rank is {}, "
                "invalid BaseSortOp {}.",
                axis,
                shape.size(),
                str());
  }
}

void BaseSortOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

int64_t BaseSortOp::getAxis() const { return axis; }

} // namespace popart
