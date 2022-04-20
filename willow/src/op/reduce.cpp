// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <vector>
#include <popart/op/reduce.hpp>
#include <popart/opserialiser.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
struct OperatorIdentifier;

ReduceOp::ReduceOp(const OperatorIdentifier &_opid,
                   const nonstd::optional<std::vector<int64_t>> &axes_,
                   const int64_t keepdims_,
                   const Op::Settings &settings_)
    : Op(_opid, settings_), axes(), keepdims(keepdims_),
      has_default_axes(!axes_) {

  if (axes_) {
    // We do have axes on construction, copy them now.
    axes = *axes_;
  }
}

std::unique_ptr<Op> ReduceOp::clone() const {
  return std::make_unique<ReduceOp>(*this);
}

void ReduceOp::setup() {
  const auto input_shape = inShape(getInIndex());

  if (has_default_axes) {
    // We didn't have axes during construction, we should reduce over ALL axes.
    axes.resize(input_shape.size());
    std::iota(axes.begin(), axes.end(), int64_t(0));
  } else {
    // Check the axes are all in the right range.
    validateReduceAxes(axes, input_shape.size(), str());
    // Normalize to positive axes.
    normalizeReduceAxes(axes, input_shape.size());
    // Sort the axes for general backend compatibility.
    std::sort(axes.begin(), axes.end());
  }

  Shape output_shape;
  output_shape.reserve(input_shape.size());
  backward_shape.clear();
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
  axes             = value;
  has_default_axes = false;

  // Sorting the axes for general backend compatibility
  std::sort(axes.begin(), axes.end());
}

void ReduceOp::setKeepDims(int64_t value) { keepdims = value; }

void ReduceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("keepdims", keepdims);
  os.appendAttribute("axes", axes);
}

// A reduce op that doesn't reduce anything can be replaced by identity
bool ReduceOp::canBeReplacedByIdentity() const {
  return (inInfo(getInIndex()).shape() == outInfo(getOutIndex()).shape());
}

const Shape &ReduceOp::backwardShape() const { return backward_shape; }

ReduceGradOp::ReduceGradOp(const AiGraphcoreOpIdV1 &opid_,
                           const ReduceOp &fwdOp,
                           const Shape &backward_shape_)
    : Op(opid_, fwdOp.getSettings()),
      outputTensorShape(fwdOp.inShape(ReduceOp::getInIndex())),
      backward_shape(backward_shape_), axes(fwdOp.getAxes()) {}

std::unique_ptr<Op> ReduceGradOp::clone() const {
  return std::make_unique<ReduceGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceOp::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &ReduceGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ReduceOp::getInIndex()}};
  return outInfo;
}

const Shape &ReduceGradOp::backwardShape() const { return backward_shape; }

void ReduceGradOp::setup() {
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outputTensorShape};
}

const std::vector<int64_t> &ReduceGradOp::getAxes() const { return axes; }

} // namespace popart
