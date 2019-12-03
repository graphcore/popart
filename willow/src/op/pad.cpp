#include <algorithm>
#include <memory>
#include <popart/op/pad.hpp>
#include <popart/op/padgrad.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

BasePadOp::BasePadOp(const OperatorIdentifier &_opid,
                     const std::vector<int64_t> &_pads,
                     float value_,
                     const std::string &_mode,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), pads(_pads), pad_value(value_), mode(_mode) {}

PadOp::PadOp(const OperatorIdentifier &_opid,
             const std::vector<int64_t> &_pads,
             float value_,
             const std::string &_mode,
             const Op::Settings &settings_)
    : BasePadOp(_opid, _pads, value_, _mode, settings_) {}

std::unique_ptr<Op> PadOp::clone() const {
  return std::make_unique<PadOp>(*this);
}

std::vector<std::unique_ptr<Op>> PadOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  if (getMode() == "constant") {
    upops.emplace_back(std::make_unique<PadGradOp>(*this));
  } else {
    // TODO : T6631 Add support for other grad op when mode is "Reflect" &
    // "Edge". May define different pad grad op classes for the different modes
    throw error("Do not support PadGradOp when mode is not \"constant\"");
  }
  return upops;
}

std::unique_ptr<Op>
PadOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::PadInplace) {
    return std::make_unique<PadInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

view::RegMap BasePadOp::fwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex != 0 || outIndex != 0) {
    throw error("Internal Logic Error in BasePadOp::fwdRegMap."
                "Received input index {} but only 0 allowed, "
                "This for Op {}, ",
                inIndex,
                str());
  }

  // add the lower padding dimensions
  auto padsCapture = pads;
  return [padsCapture](const view::Region &r) {
    view::LowBounds out_lb = r.getLower();
    view::UppBounds out_ub = r.getUpper();
    for (int i = 0; i < padsCapture.size() / 2; ++i) {
      out_lb[i] += std::max<int64_t>(padsCapture[i], 0);
      out_ub[i] += std::max<int64_t>(padsCapture[i], 0);
    }
    return view::Regions(1, view::Region(out_lb, out_ub));
  };
}

view::RegMap BasePadOp::bwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex != 0 || outIndex != 0) {
    throw error("Internal Logic Error in BasePadOp::bwdRegMap."
                "Received input index {} but only 0 allowed, "
                "This for Op {}, ",
                inIndex,
                str());
  }

  // add the lower padding dimensions
  auto valReg      = valueRegion();
  auto padsCapture = pads;
  return [valReg, padsCapture](const view::Region &rIn) {
    auto r                 = rIn.intersect(valReg);
    view::LowBounds out_lb = r.getLower();
    view::UppBounds out_ub = r.getUpper();
    for (int i = 0; i < padsCapture.size() / 2; ++i) {
      out_lb[i] -= std::max<int64_t>(padsCapture[i], 0);
      out_ub[i] -= std::max<int64_t>(padsCapture[i], 0);
    }
    return view::Regions(1, view::Region(out_lb, out_ub));
  };
}

PadInplaceOp::PadInplaceOp(const PadOp &padOp)
    : BasePadOp(Onnx::CustomOperators::PadInplace,
                padOp.getPads(),
                padOp.getPadValue(),
                padOp.getMode(),
                padOp.getSettings()) {}

view::Regions PadInplaceOp::modifies(InIndex index) const {
  return uses(index);
}
view::Regions PadInplaceOp::aliases(InIndex in, OutIndex) const {
  return uses(in);
}
view::Regions PadInplaceOp::uses(InIndex index) const {
  if (index == 0) {
    return {view::Region::getFull(inShape(index))};
  }
  throw error("ILE : invalid InIndex to PadInplaceOp::uses");
}

std::unique_ptr<Op> PadInplaceOp::clone() const {
  return std::make_unique<PadInplaceOp>(*this);
}

std::vector<std::tuple<OperatorIdentifier, float>>
PadOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::PadInplace, 20}};
}

void BasePadOp::setup() {

  int tRank = inRank(getInIndex());
  if (getPads().size() != 2 * tRank) {
    throw error("Tensor rank not half padding size");
  }

  Shape outShape(tRank, 0);
  for (int i = 0; i < tRank; ++i) {
    outShape[i] = inInfo(getInIndex()).dim(i) + pads[i] + pads[i + tRank];
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outShape};
}

bool BasePadOp::padSizeZero() const {
  return std::all_of(
      pads.cbegin(), pads.cend(), [](int64_t p) { return p == 0; });
}

const std::vector<int64_t> &BasePadOp::getPads() const { return pads; }

float BasePadOp::getPadValue() const { return pad_value; }

const std::string &BasePadOp::getMode() const { return mode; }

void BasePadOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);

  os.appendAttribute("pads", pads);
  os.appendAttribute("value", pad_value);
  os.appendAttribute("mode", mode);
}

view::Region BasePadOp::valueRegion() const {
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

std::vector<int64_t> BasePadOp::padDimensions() const {
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
  return std::make_unique<PadGradOp>(*this);
}

const std::vector<GradInOutMapper> &PadGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), BasePadOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &PadGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), BasePadOp::getInIndex()}};

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

// The BasePadOp provides pad information for each axis.
// The default of the SliceOp when axes is blank is to assume start & end
// for all axes
std::vector<int64_t> PadGradOp::calculateAxes(const PadOp &) {
  std::vector<int64_t> axes = {};
  return axes;
}

namespace {

static OpDefinition::DataTypes T  = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::INT64};

static OpDefinition
    padOpV2Def({OpDefinition::Inputs({{"data", T}}),
                OpDefinition::Outputs({{"output", T}}),
                OpDefinition::Attributes({{"mode", {"constant|reflect|edge"}},
                                          {"pads", {"*"}},
                                          {"value", {"*"}}})});

static OpDefinition padOpV11Def({OpDefinition::Inputs({
                                     {"data", T},
                                     {"pads", T1, true},
                                     {"constant_value", T, true},
                                 }),
                                 OpDefinition::Outputs({{"output", T}}),
                                 OpDefinition::Attributes({
                                     {"mode", {"constant|reflect|edge"}},
                                 })});

static OpCreator<PadOp> padCreator(
    OpDefinitions({{Onnx::Operators::Pad_2, padOpV2Def},
                   {Onnx::Operators::Pad_11, padOpV11Def}}),
    //{Onnx::Operators::Pad_2, Onnx::Operators::Pad_11},
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
} // namespace

} // namespace popart
