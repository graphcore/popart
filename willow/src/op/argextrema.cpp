#include <algorithm>
#include <vector>

#include <memory>
#include <popart/op/argextrema.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

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

void ArgExtremaOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
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

  if (shape.size() <= axis) {
    throw error("Cannot compute ArgExtremaOp on axis {} when input rank is {}, "
                "invalid ArgExtremaOp {}.",
                axis,
                shape.size(),
                str());
  }
}

int64_t ArgExtremaOp::getKeepDims() const { return keepdims; }

int64_t ArgExtremaOp::getAxis() const { return axis; }

} // namespace popart
