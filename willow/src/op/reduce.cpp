#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/reduce.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReduceOp::ReduceOp(const OperatorIdentifier &_opid,
                   const std::vector<int64_t> &axes_,
                   const int64_t keepdims_,
                   const Op::Settings &settings_)
    : Op(_opid, settings_), axes(axes_), keepdims(keepdims_) {

  // Sorting the axes for general backend compatibility
  std::sort(axes.begin(), axes.end());
}

std::unique_ptr<Op> ReduceOp::clone() const {
  return make_unique<ReduceOp>(*this);
}

void ReduceOp::setup() {
  const auto input_shape = inShape(getInIndex());

  Shape output_shape;
  output_shape.reserve(input_shape.size());
  backward_shape.reserve(input_shape.size());

  for (int i = 0; i < input_shape.size(); ++i) {
    if (!std::count(axes.begin(), axes.end(), i)) {
      output_shape.push_back(input_shape[i]);
      backward_shape.push_back(input_shape[i]);
    } else if (keepdims) {
      output_shape.push_back(1);
      backward_shape.push_back(1);
    } else {
      backward_shape.push_back(1);
    }
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), output_shape};
}

const std::vector<int64_t> &ReduceOp::getAxes() const { return axes; }

bool ReduceOp::getKeepDims() const { return keepdims; }

void ReduceOp::setAxes(std::vector<int64_t> value) {
  axes = value;

  // Sorting the axes for general backend compatibility
  std::sort(axes.begin(), axes.end());
}

void ReduceOp::setKeepDims(int64_t value) { keepdims = value; }

void ReduceOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("keepdims", keepdims);
  os.appendAttribute("axes", axes);
}

// A reduce op that doesn't reduce anything can be replaced by identity
bool ReduceOp::canBeReplacedByIdentity() {
  return (inInfo(getInIndex()).shape() == outInfo(getOutIndex()).shape());
}

ReduceGradOp::ReduceGradOp(const AiGraphcoreOpIdV1 &opid_,
                           const ReduceOp &fwdOp,
                           const Shape &backward_shape_)
    : Op(opid_, fwdOp.getSettings()),
      outputTensorInfo(fwdOp.inInfo(ReduceOp::getInIndex())),
      backward_shape(backward_shape_) {}

std::unique_ptr<Op> ReduceGradOp::clone() const {
  return make_unique<ReduceGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &ReduceGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ReduceOp::getInIndex()}};
  return outInfo;
}

const Shape &ReduceGradOp::backwardShape() const { return backward_shape; }

void ReduceGradOp::setup() { outInfo(getOutIndex()) = outputTensorInfo; }

} // namespace poponnx
