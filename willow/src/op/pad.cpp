#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

PadOp::PadOp(const OperatorIdentifier &_opid,
             Ir *_ir,
             const std::string &name,
             const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {
  nAtts.set(pads, "pads");
}

std::unique_ptr<Op> PadOp::clone() const { return make_unique<PadOp>(*this); }

void PadOp::setup() {

  int tRank = inRank(getInIndex());
  if (pads.size() != 2 * tRank) {
    throw error("Tensor rank not half padding size");
  }

  Shape outShape(tRank, 0);
  for (int i = 0; i < tRank; ++i) {
    outShape[i] = inInfo(getInIndex()).dim(i) + pads[i] + pads[i + tRank];
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outShape};
}

bool PadOp::padSizeZero() const {
  return std::all_of(
      pads.cbegin(), pads.cend(), [](int64_t p) { return p == 0; });
}

namespace {
static OpCreator<PadOp> padCreator(Onnx::Operators::Pad);
}

} // namespace poponnx
