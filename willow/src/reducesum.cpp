#include <algorithm>

#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/reducesum.hpp>
#include <poponnx/tensor.hpp>

#include <iostream>

namespace willow {

ReduceSumOp::ReduceSumOp(const OpConstructorBundle &bundle) : Op(bundle) {
  nAtts.setIfPresent(axes, "axes");
  nAtts.setIfPresent(keepdims, "keepdims");

  // Sorting the axes for general backend compatibility
  std::sort(axes.begin(), axes.end());
}

ReduceSumOp::ReduceSumOp(const OpConstructorBundle &bundle,
                         const std::vector<int64_t> &axes_,
                         int64_t keepdims_)
    : Op(bundle), axes(axes_), keepdims(keepdims_) {

  // Sorting the axes for general backend compatibility
  std::sort(axes.begin(), axes.end());
}

ReduceSumOp::ReduceSumOp(const onnx::NodeProto &node, Ir *_pir)
    : Op(node, _pir), keepdims(0) {
  nAtts.setIfPresent(axes, "axes");
  nAtts.setIfPresent(keepdims, "keepdims");

  // Sorting the axes for general backend compatibility
  std::sort(axes.begin(), axes.end());
}

std::unique_ptr<Op> ReduceSumOp::clone() const {
  return std::unique_ptr<Op>(new ReduceSumOp(*this));
}

std::vector<std::unique_ptr<Op>> ReduceSumOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(new ReduceSumGradOp(this, backward_shape));
  return result;
}

void ReduceSumOp::setup() {
  const auto input_shape = input.tensor(0)->info.shape();

  std::vector<int64_t> output_shape;
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

  output.tensor(0)->info = {input.tensor(0)->info.dataType(), output_shape};
}

const std::vector<int64_t> &ReduceSumOp::getAxes() const { return axes; }

bool ReduceSumOp::getKeepDims() const { return keepdims; }

ReduceSumGradOp::ReduceSumGradOp(ReduceSumOp *fwdOp,
                                 const std::vector<int64_t> &backward_shape_)
    : Op({"ReduceSumGrad", fwdOp->pir, {}, getPoponnxDomain()}),
      outputTensorInfo(fwdOp->input.tensor(0)->info),
      backward_shape(backward_shape_) {}

std::unique_ptr<Op> ReduceSumGradOp::clone() const {
  return std::unique_ptr<Op>(new ReduceSumGradOp(*this));
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

const std::vector<int64_t> &ReduceSumGradOp::backwardShape() const {
  return backward_shape;
}

void ReduceSumGradOp::setup() { output.tensor(0)->info = outputTensorInfo; }

} // namespace willow
