#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

PadOp::PadOp(const OperatorIdentifier &_opid,
             const std::vector<int64_t> &_pads,
             float value_,
             const std::string &_mode,
             const Op::Settings &settings_)
    : Op(_opid, settings_), pads(_pads), pad_value(value_), mode(_mode) {}

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

void PadOp::appendAttributes(std::stringstream &ss,
                             const std::string &tab) const {
  Op::appendAttributes(ss, tab);

  appendAttribute(ss, tab, "pads", pads);
  appendAttribute(ss, tab, "value", pad_value);
  appendAttribute(ss, tab, "mode", mode);
}
namespace {
static OpCreator<PadOp> padCreator(
    Onnx::Operators::Pad_2,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      std::vector<int64_t> pads = attr.getAttribute<Attributes::Ints>("pads");
      float value = attr.getAttribute<Attributes::Float>("value", 0.0);
      std::string mode =
          attr.getAttribute<Attributes::String>("mode", "constant");

      return std::unique_ptr<Op>(new PadOp(_opid, pads, value, mode, settings));
    },
    true);
}

} // namespace poponnx
