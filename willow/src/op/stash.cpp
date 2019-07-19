#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/stash.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensornames.hpp>

namespace poponnx {

StashOp::StashOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> StashOp::clone() const {
  return std::make_unique<StashOp>(*this);
}

void StashOp::setup() {
  Shape output_shape = inShape(getInIndex());
  output_shape.insert(output_shape.begin(), getStashSize());

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), output_shape};
}

int64_t StashOp::getStashSize() {
  int64_t vGraphId = getVirtualGraphId();
  int64_t numIPUs = static_cast<int64_t>(getIr().getDeviceInfo()->getNumIpus());

  if (vGraphId == numIPUs - 1) {
    throw error("There should be no stashes on the final IPU.");
  }
  return 2 * (numIPUs - vGraphId) - 1;
}

TensorId StashOp::getStashedTensorId() const {
  return reservedStashedPrefix() + inId(getInIndex());
}

namespace {
static OpCreator<StashOp> StashOpCreator(Onnx::CustomOperators::Stash);

} // namespace

} // namespace poponnx
