#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/padgrad.hpp>
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

std::vector<std::unique_ptr<Op>> PadOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  if (mode == "constant") {
    upops.emplace_back(make_unique<PadGradOp>(*this));
  } else {
    // TODO : T6631 Add support for other grad op when mode is "Reflect" &
    // "Edge". May define different pad grad op classes for the different modes
    throw error("Do not support PadGradOp when mode is not \"constant\"");
  }
  return upops;
}

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

view::Region PadOp::valueRegion() const {
  std::vector<int64_t> lower(pads.size() / 2);
  std::vector<int64_t> upper(pads.size() / 2);

  const auto shape = outShape(getOutIndex());

  for (int i = 0; i < pads.size() / 2; ++i) {
    lower[i] = std::max<int64_t>(pads[i], 0);
  }

  for (int i = 0; i < pads.size() / 2; ++i) {
    auto idx = (pads.size() / 2) + i;
    upper[i] = shape[i] - std::max<int64_t>(pads[idx], 0);
  }

  return {lower, upper};
}

std::vector<int64_t> PadOp::padDimensions() const {
  std::set<int64_t> dimensions;

  for (int i = 0; i < pads.size() / 2; ++i) {
    if (pads[i] != 0) {
      dimensions.insert(i);
    }
  }

  for (int i = 0; i < pads.size() / 2; ++i) {
    if (pads[(pads.size() / 2) + i] != 0) {
      dimensions.insert(i);
    }
  }

  return {dimensions.begin(), dimensions.end()};
}

// If pad has no padding it can be replaced by identity
bool PadOp::canBeReplacedByIdentity() { return padSizeZero(); }

PadGradOp::PadGradOp(const PadOp &fwdOp)
    : SliceOp(Onnx::GradOperators::PadGrad,
              calculateStarts(fwdOp),
              calculateEnds(fwdOp),
              calculateAxes(fwdOp),
              fwdOp.getSettings()) {}

std::unique_ptr<Op> PadGradOp::clone() const {
  return make_unique<PadGradOp>(*this);
}

const std::vector<GradInOutMapper> &PadGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), PadOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &PadGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), PadOp::getInIndex()}};

  return outInfo;
}

std::vector<int64_t> PadGradOp::calculateStarts(const PadOp &padOp) {

  std::vector<int64_t> starts(padOp.getPads().size() / 2);

  // Set the starts to the 'index' after the 'begin' padding
  for (int i = 0; i < padOp.getPads().size() / 2; ++i) {
    starts[i] = (padOp.getPads()[i]);
  }

  return starts;
}

std::vector<int64_t> PadGradOp::calculateEnds(const PadOp &padOp) {
  std::vector<int64_t> ends(padOp.getPads().size() / 2);

  // Set the ends to the 'index' where the original input would have ended
  for (int i = 0; i < padOp.getPads().size() / 2; ++i) {
    ends[i] = padOp.getPads()[i] + padOp.inShape(padOp.getInIndex())[i];
  }

  return ends;
}

// The PadOp provides pad information for each axis.
// The default of the SliceOp when axes is blank is to assume start & end
// for all axes
std::vector<int64_t> PadGradOp::calculateAxes(const PadOp &) {
  std::vector<int64_t> axes = {};
  return axes;
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
