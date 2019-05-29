#include <poponnx/op/basesort.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

BaseSortOp::BaseSortOp(const OperatorIdentifier &_opid,
                       int64_t axis_,
                       const Op::Settings &settings)
    : Op(_opid, settings), axis(axis_) {}

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

void BaseSortOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
}

int64_t BaseSortOp::getAxis() const { return axis; }

} // namespace poponnx
