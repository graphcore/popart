#include <algorithm>
#include <vector>

#include <poponnx/makeunique.hpp>
#include <poponnx/op/argextrema.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ArgExtremaOp::ArgExtremaOp(const OperatorIdentifier &_opid,
                           int64_t axis_,
                           int64_t keepdims_,
                           const Op::Settings &settings)
    : Op(_opid, settings), axis(axis_), keepdims(keepdims_) {}

std::unique_ptr<Op> ArgExtremaOp::clone() const {
  return make_unique<ArgExtremaOp>(*this);
}

void ArgExtremaOp::setup() {
  auto shape = inShape(getInIndex());

  if (shape.size() == 0) {
    throw error("ArgExtremaOp input rank must be greater than 0");
  }

  if (shape.size() <= axis) {
    throw error("Cannot compute ArgExtremaOp on axis {} when input rank is {}",
                axis,
                shape.size());
  }

  shape[axis] = 1;

  if (keepdims == 0) {
    shape.erase(shape.begin() + axis);
  }

  outInfo(getOutIndex()) = TensorInfo(DataType::INT32, shape);
}

void ArgExtremaOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("keepdims", keepdims);
}

int64_t ArgExtremaOp::getAxis() const { return axis; }

int64_t ArgExtremaOp::getKeepDims() const { return keepdims; }

} // namespace poponnx
