#include <limits>
#include <memory>
#include <poponnx/op/gradientaccl.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/region.hpp>
#include <poponnx/tensornames.hpp>

namespace poponnx {
GradientAcclOp::GradientAcclOp(const OperatorIdentifier &_opid,
                               const Op::Settings &settings_)
    : Op(_opid, settings_) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void GradientAcclOp::setup() {
  outInfo(getAcclOutIndex()) = inInfo(getAcclInIndex());
}

// Modifies the whole of the Var Tensor
view::Region GradientAcclOp::modifies(InIndex index) const {
  return aliases(index);
}

view::Region GradientAcclOp::aliases(InIndex index) const {
  if (index == GradientAcclOp::getAcclInIndex()) {
    return view::Region::getFull(inShape(index));
  } else {
    return view::Region::getEmpty(inRank(index));
  }
}

std::unique_ptr<Op> GradientAcclOp::clone() const {
  return std::make_unique<GradientAcclOp>(*this);
}

ResetAcclOp::ResetAcclOp(const OperatorIdentifier &_opid,
                         const Op::Settings &settings_)
    : Op(_opid, settings_) {
  // Should be the final operation in the graph
  priority = std::numeric_limits<double>::lowest();
}

void ResetAcclOp::setup() {
  outInfo(getAcclOutIndex()) = inInfo(getAcclInIndex());
}

std::unique_ptr<Op> ResetAcclOp::clone() const {
  return std::make_unique<ResetAcclOp>(*this);
}
view::Region ResetAcclOp::aliases(InIndex index) const {
  return view::Region::getFull(inShape(index));
}
// Modifies is the same as aliases
view::Region ResetAcclOp::modifies(InIndex index) const {
  return aliases(index);
}

// Have intentionally not added the these Ops to the OpManager. They
// need to be explicitly created as part of the gradient_accumulation transform

} // namespace poponnx
