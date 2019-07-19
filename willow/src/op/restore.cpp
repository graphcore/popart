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

view::Region RestoreOp::aliases(InIndex index) const {
  if (index == getActToRestoreInIndex()) {
    return view::Region::getFull(inShape(index));
  } else {
    return view::Region::getEmpty(inRank(index));
  }
}

// Modifies is the same as aliases
view::Region RestoreOp::modifies(InIndex index) const { return aliases(index); }

TensorId RestoreOp::getRestoredTensorId() const {
  return reservedRestoredPrefix() + inId(getActToRestoreInIndex());
}

namespace {
static OpCreator<RestoreOp> RestoreOpCreator(Onnx::CustomOperators::Restore);

} // namespace

} // namespace poponnx
