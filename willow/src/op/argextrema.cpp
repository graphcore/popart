// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op/argextrema.hpp>
#include <popart/opserialiser.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

ArgExtremaOp::ArgExtremaOp(const OperatorIdentifier &opid_,
                           int64_t axis_,
                           int64_t keepdims_,
                           const Op::Settings &settings_)
    : Op(opid_, settings_), keepdims(keepdims_), axis(axis_) {}

std::unique_ptr<Op> ArgExtremaOp::clone() const {
  return std::make_unique<ArgExtremaOp>(*this);
}

void ArgExtremaOp::setup() {

  validateAxis();

  auto shape       = inShape(getInIndex());
  shape[getAxis()] = 1;
  if (keepdims == 0) {
    shape.erase(shape.begin() + getAxis());
  }

  outInfo(getOutIndex()) = TensorInfo(DataType::INT32, shape);
}

void ArgExtremaOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("keepdims", keepdims);
  os.appendAttribute("axis", axis);
}

void ArgExtremaOp::validateAxis() const {
  auto shape = inShape(getInIndex());

  if (shape.size() == 0) {
    throw error("ArgExtremaOp input rank must be greater than 0, invalid "
                "ArgExtremaOp {}.",
                str());
  }

  if (shape.size() <= getAxis()) {
    throw error("Cannot compute ArgExtremaOp on axis {} when input rank is {}, "
                "invalid ArgExtremaOp {}.",
                axis,
                shape.size(),
                str());
  }

  // From the onnx spec:
  //   Accepted range is [-r, r-1] where r = rank(data).
  if (axis > static_cast<int64_t>(shape.size()) - 1 ||
      axis < -static_cast<int64_t>(shape.size())) {
    throw error("Axis {} is out of acceptable range [{}, {}]",
                axis,
                -static_cast<int64_t>(shape.size()),
                shape.size() - 1);
  }
}

int64_t ArgExtremaOp::getKeepDims() const { return keepdims; }

int64_t ArgExtremaOp::getAxis() const {
  // Onnx 11 supports negative axis indexing for argmin and argmax.
  if (axis >= 0) {
    return axis;
  } else {
    return inInfo(getInIndex()).rank() + axis;
  }
}

} // namespace popart
