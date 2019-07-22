#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/restore.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensornames.hpp>

namespace poponnx {

RestoreOp::RestoreOp(const OperatorIdentifier &_opid,
                     const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> RestoreOp::clone() const {
  return std::make_unique<RestoreOp>(*this);
}

void RestoreOp::setup() {
  outInfo(getRestoredActOutIndex()) = inInfo(getActToRestoreInIndex());
}

TensorId RestoreOp::getRestoredTensorId() const {
  return reservedRestoredPrefix() + inId(getActToRestoreInIndex());
}

RestoreInplaceOp::RestoreInplaceOp(const OperatorIdentifier &_opid,
                                   const Op::Settings &settings_)
    : RestoreOp(_opid, settings_) {}

std::unique_ptr<Op> RestoreInplaceOp::clone() const {
  return std::make_unique<RestoreInplaceOp>(*this);
}

view::Region RestoreInplaceOp::aliases(InIndex index) const {
  if (index == getActToRestoreInIndex()) {
    return view::Region::getFull(inShape(index));
  } else {
    return view::Region::getEmpty(inRank(index));
  }
}

// Modifies is the same as aliases
view::Region RestoreInplaceOp::modifies(InIndex index) const {
  return aliases(index);
}

namespace {
static OpCreator<RestoreOp> RestoreOpCreator(Onnx::CustomOperators::Restore);
static OpCreator<RestoreOp>
    RestoreInplaceOpCreator(Onnx::CustomOperators::RestoreInplace);

} // namespace

} // namespace poponnx
