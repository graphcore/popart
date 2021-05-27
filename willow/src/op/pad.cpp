// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <aliasmodel.hpp>
#include <memory>
#include <popart/graph.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/padgrad.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

BasePadOp::BasePadOp(const OperatorIdentifier &_opid,
                     const std::vector<int64_t> &_pads,
                     const std::vector<unsigned> &_flips,
                     float value_,
                     const std::string &_mode,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), pads(_pads), flips(_flips), pad_value(value_),
      mode(_mode) {}

BasePadOutplaceOp::BasePadOutplaceOp(const OperatorIdentifier &_opid,
                                     const std::vector<int64_t> &_pads,
                                     const std::vector<unsigned> &_flips,
                                     float value_,
                                     const std::string &_mode,
                                     const Op::Settings &settings_)
    : BasePadOp(_opid, _pads, _flips, value_, _mode, settings_) {}

PadOp::PadOp(const OperatorIdentifier &_opid,
             const std::vector<int64_t> &_pads,
             const std::vector<unsigned> &_flips,
             float value_,
             const std::string &_mode,
             const Op::Settings &settings_)
    : BasePadOutplaceOp(_opid, _pads, _flips, value_, _mode, settings_) {}

// check pads are even, and check than rank agrees with input tensor
void BasePadOp::runtimeConfirmShapes() const {
  if (pads.size() != 2 * inInfo(getInIndex()).rank()) {
    std::ostringstream oss;
    oss << "Failure in BasePadOp::runtimeConfirmShapes, for " << str()
        << ". This with pads of size " << pads.size()
        << ", and input tensor of rank " << inInfo(getInIndex()).rank()
        << ". Expected pads to be exactly 2x the unput Tensor's rank. ";
    throw error(oss.str());
  }
}

std::vector<std::ptrdiff_t> BasePadOp::getPadRange(size_t startIndex) const {
  runtimeConfirmShapes();
  std::vector<std::ptrdiff_t> subPadding;
  subPadding.reserve(getRank());
  for (auto iter = std::next(pads.cbegin(), startIndex);
       iter != std::next(pads.cbegin(), getRank() + startIndex);
       ++iter) {
    subPadding.push_back(*iter);
  }
  return subPadding;
}

std::vector<Slice> BasePadOp::getSlices() const {
  const auto lowerPadding = getLowerPadding();
  const auto upperPadding = getUpperPadding();
  std::vector<Slice> slices;
  slices.reserve(getRank());
  for (auto i = 0ULL; i < getRank(); ++i) {
    slices.push_back(
        {lowerPadding[i], upperPadding[i], static_cast<int64_t>(i)});
  }
  return slices;
}

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
    throw error("At this time, PopART does not support PadGradOp when mode is "
                "not \"constant\".");
  }
  return upops;
}

void BasePadOp::growAliasModel(AliasModel &m) const {

  runtimeConfirmShapes();

  const auto in0 = m.getPoprithmsTensorId(inId(0));

  auto toPad = doesAlias() ? m.g.aliasGate({in0}, 0) : m.g.aliasGate({in0});
  m.insertOp(toPad.opId(), id);

  auto lowerPadding = getLowerPadding();
  auto upperPadding = getUpperPadding();

  for (auto &x : lowerPadding) {
    x = std::max(x, int64_t(0));
  }

  for (auto &x : upperPadding) {
    x = std::max(x, int64_t(0));
  }

  const auto out0 = m.g.pad(toPad,
                            {lowerPadding, upperPadding},
                            /*paddingIsParallelWriteable*/ false);

  m.insertTensor(out0, *outTensor(0));
}

void BasePadOutplaceOp::setProposal(
    poprithms::memory::inplace::Proposal &proposal,
    const AliasModel &aliaser,
    OperatorIdentifier opId) const {
  proposal = {aliaser.getGate(id), 0};
}

std::unique_ptr<Op> BasePadOutplaceOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::PadInplace) {
    return std::make_unique<PadInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

void PadOp::connectInTensor(InIndex inIndex, TensorId tenId) {
  // Ignore all but the first input
  if (inIndex == getInIndex()) {
    Op::connectInTensor(inIndex, tenId);
  }
}

view::RegMap BasePadOp::fwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex != 0 || outIndex != 0) {
    throw internal_error("[BasePadOp::fwdRegMap] "
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
    throw internal_error("[BasePadOp::bwdRegMap] "
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

PadInplaceOp::PadInplaceOp(const BasePadOutplaceOp &padOp)
    : BasePadOp(Onnx::CustomOperators::PadInplace,
                padOp.getPads(),
                padOp.getFlips(),
                padOp.getPadValue(),
                padOp.getMode(),
                padOp.getSettings()) {}

view::Regions PadInplaceOp::aliases(InIndex in, OutIndex) const {
  return uses(in);
}
view::Regions PadInplaceOp::uses(InIndex index) const {
  if (index == 0) {
    return {view::Region::getFull(inShape(index))};
  }
  throw internal_error("invalid InIndex to PadInplaceOp::uses");
}

std::unique_ptr<Op> PadInplaceOp::clone() const {
  return std::make_unique<PadInplaceOp>(*this);
}

std::vector<std::tuple<OperatorIdentifier, float>>
BasePadOutplaceOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::PadInplace, 1.0}};
}

void BasePadOp::setup() {

  runtimeConfirmShapes();

  Shape outShape(getRank(), 0);
  for (auto i = 0; i < getRank(); ++i) {
    outShape[i] = inInfo(getInIndex()).dim(i) + pads[i] + pads[i + getRank()];
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outShape};
}

bool BasePadOp::padSizeZero() const {
  return std::all_of(
      pads.cbegin(), pads.cend(), [](int64_t p) { return p == 0; });
}

void BasePadOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("pads", pads);
  os.appendAttribute("value", pad_value);
  os.appendAttribute("mode", mode);
}

view::Region BasePadOp::valueRegion() const {
  std::vector<int64_t> lower(getRank());
  std::vector<int64_t> upper(getRank());

  const auto shape = outShape(getOutIndex());

  for (int i = 0; i < getRank(); ++i) {
    lower[i] = std::max<int64_t>(pads[i], 0);
  }

  for (int i = 0; i < getRank(); ++i) {
    auto idx = getRank() + i;
    upper[i] = shape[i] - std::max<int64_t>(pads[idx], 0);
  }

  return {lower, upper};
}

std::vector<int64_t> BasePadOp::padDimensions() const {
  std::vector<int64_t> dimensions;
  for (int i = 0; i < getRank(); ++i) {
    if (getLowerPadding(i) != 0 || getUpperPadding(i) != 0) {
      dimensions.push_back(i);
    }
  }
  return dimensions;
}

PadGradOp::PadGradOp(const PadOp &fwdOp)
    : SliceOp(Onnx::GradOperators::PadGrad,
              calculateStarts(fwdOp),
              calculateEnds(fwdOp),
              calculateAxes(fwdOp),
              {}, // empty steps
              fwdOp.getSettings()) {}

std::unique_ptr<Op> PadGradOp::clone() const {
  return std::make_unique<PadGradOp>(*this);
}

const std::vector<GradInOutMapper> &PadGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), BasePadOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &PadGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), BasePadOp::getInIndex()}};

  return outInfo;
}

std::vector<int64_t> PadGradOp::calculateStarts(const PadOp &padOp) {

  std::vector<int64_t> starts(padOp.getRank());

  // Set the starts to the 'index' after the 'begin' padding
  for (int i = 0; i < padOp.getRank(); ++i) {
    starts[i] = (padOp.getLowerPadding(i));
  }

  return starts;
}

std::vector<int64_t> PadGradOp::calculateEnds(const PadOp &padOp) {
  std::vector<int64_t> ends(padOp.getRank());

  // Set the ends to the 'index' where the original input would have ended
  for (int i = 0; i < padOp.getRank(); ++i) {
    ends[i] = padOp.getLowerPadding(i) + padOp.inShape(padOp.getInIndex())[i];
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

static OpCreator<PadOp> pad2Creator(
    OpDefinitions({{Onnx::Operators::Pad_2, padOpV2Def}}),
    [](const OpCreatorInfo &info) {
      std::vector<int64_t> pads =
          info.attributes.getAttribute<Attributes::Ints>("pads");
      float value =
          info.attributes.getAttribute<Attributes::Float>("value", 0.0);
      std::string mode =
          info.attributes.getAttribute<Attributes::String>("mode", "constant");

      return std::unique_ptr<Op>(
          new PadOp(info.opid, pads, {}, value, mode, info.settings));
    },
    true);

static OpCreator<PadOp> pad11Creator(
    OpDefinitions({{Onnx::Operators::Pad_11, padOpV11Def}}),
    [](const OpCreatorInfo &info, Graph &graph) {
      int padsInputIndex  = 1;
      int valueInputIndex = 2;

      std::vector<int64_t> pads = info.getInputData<int64_t>(padsInputIndex);
      float value = info.getInputScalarValue<float>(valueInputIndex, 0.0);

      std::string mode = "constant";
      if (info.attributes.hasAttribute("mode")) {
        mode = info.attributes.getAttribute<Attributes::String>("mode");
      }

      // Create the op in the graph.
      Op *op = graph.createOp<PadOp>(
          info.opid, pads, std::vector<unsigned>{}, value, mode, info.settings);

      // Connect only the first of the two inputs.
      op->connectInTensor(PadOp::getInIndex(), info.getInputIds().at(0));
      op->createAndConnectOutTensor(PadOp::getOutIndex(),
                                    info.getOutputIds().at(0));

      return op;
    },
    true);
} // namespace

} // namespace popart
