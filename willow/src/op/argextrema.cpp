#include <algorithm>
#include <vector>

#include <poponnx/makeunique.hpp>
#include <poponnx/op/argextrema.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ArgExtremaOp::ArgExtremaOp(const OperatorIdentifier &opid_,
                           int64_t axis_,
                           int64_t keepdims_,
                           const Op::Settings &settings)
    : BaseSortOp(opid_, axis_, settings), keepdims(keepdims_) {}

std::unique_ptr<Op> ArgExtremaOp::clone() const {
  return make_unique<ArgExtremaOp>(*this);
}

void ArgExtremaOp::setup() {

  validateAxis();

  auto shape       = inShape(BaseSortOp::getInIndex());
  shape[getAxis()] = 1;
  if (keepdims == 0) {
    shape.erase(shape.begin() + getAxis());
  }

  outInfo(getOutIndex()) = TensorInfo(DataType::INT32, shape);
}

void ArgExtremaOp::appendAttributes(OpSerialiserBase &os) const {
  BaseSortOp::appendAttributes(os);
  os.appendAttribute("keepdims", keepdims);
}

int64_t ArgExtremaOp::getKeepDims() const { return keepdims; }

} // namespace poponnx
