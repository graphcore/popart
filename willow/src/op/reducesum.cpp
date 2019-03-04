#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReduceSumOp::ReduceSumOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &axes_,
                         const int64_t keepdims_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), axes(axes_), keepdims(keepdims_) {

  // Sorting the axes for general backend compatibility
  std::sort(axes.begin(), axes.end());
}

std::unique_ptr<Op> ReduceSumOp::clone() const {
  return make_unique<ReduceSumOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceSumOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(make_unique<ReduceSumGradOp>(*this, backward_shape));
  return result;
}

void ReduceSumOp::setup() {
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

const std::vector<int64_t> &ReduceSumOp::getAxes() const { return axes; }

bool ReduceSumOp::getKeepDims() const { return keepdims; }

void ReduceSumOp::setAxes(std::vector<int64_t> value) {
  axes = value;

  // Sorting the axes for general backend compatibility
  std::sort(axes.begin(), axes.end());
}

void ReduceSumOp::setKeepDims(int64_t value) { keepdims = value; }

void ReduceSumOp::appendAttributes(std::stringstream &ss,
                                   const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "keepdims", keepdims);
  appendAttribute(ss, tab, "axes", axes);
}

// A reducesum op that doesn't reduce anything can be replaced by
// identity
bool ReduceSumOp::canBeReplacedByIdentity() {
  return (inInfo(getInIndex()).shape() == outInfo(getOutIndex()).shape());
}

ReduceSumGradOp::ReduceSumGradOp(const ReduceSumOp &fwdOp,
                                 const Shape &backward_shape_)
    : Op(Onnx::GradOperators::ReduceSumGrad, fwdOp.getSettings()),
      outputTensorInfo(fwdOp.inInfo(ReduceSumOp::getInIndex())),
      backward_shape(backward_shape_) {}

std::unique_ptr<Op> ReduceSumGradOp::clone() const {
  return make_unique<ReduceSumGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceSumGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &ReduceSumGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};

  return outInfo;
}

const Shape &ReduceSumGradOp::backwardShape() const { return backward_shape; }

void ReduceSumGradOp::setup() { outInfo(getOutIndex()) = outputTensorInfo; }

namespace {
// @SL@ the new factory method for the reduceSum op will get the attributes from
// the model and pass them to the constructor of the OP
static OpCreator<ReduceSumOp> reduceSumOpCreator(
    Onnx::Operators::ReduceSum_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new ReduceSumOp(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace poponnx
