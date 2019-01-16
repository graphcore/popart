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
  pad_value = 0.0;
  nAtts.setIfPresent(pad_value, "value");
  nAtts.set(pads, "pads");

  nAtts.setIfPresent(mode, "mode");
  if (mode.empty()) {
    mode = "constant";
  }
}

PadOp::PadOp(const OperatorIdentifier &_opid,
             Ir *_ir,
             const std::vector<int64_t> _pads,
             float _pad_value,
             std::string _mode)
    : Op(_opid, _ir), pads(_pads), pad_value(_pad_value), mode(_mode) {}

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

const std::vector<int64_t> &PadOp::getPads() const { return pads; }

float PadOp::getPadValue() const { return pad_value; }

const std::string &PadOp::getMode() const { return mode; }

namespace {
static OpCreator<PadOp> padCreator(Onnx::Operators::Pad_2);
}

} // namespace poponnx
