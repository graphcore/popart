// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/histogram.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

std::unique_ptr<Op> HistogramOp::clone() const {
  return std::make_unique<HistogramOp>(*this);
}

HistogramOp::HistogramOp(const OperatorIdentifier &_opid,
                         const std::vector<float> &levels_,
                         const bool absoluteOfInput_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), levels(levels_), absoluteOfInput(absoluteOfInput_) {

  // The bin edges must appear in ascending order
  std::sort(levels.begin(), levels.end());
}

namespace {
static std::vector<DataType> supportedTypes = {DataType::FLOAT,
                                               DataType::FLOAT16};
static DataType outputType                  = DataType::UINT32;
} // namespace

void HistogramOp::setup() {
  auto inType = inInfo(getInIndex()).dataType();
  if (std::find(supportedTypes.begin(), supportedTypes.end(), inType) ==
      supportedTypes.end()) {
    throw error("HistogramOp {} input type {} is not supported.",
                str(),
                inInfo(getInIndex()).data_type());
  }

  outInfo(getOutIndex()) = {outputType,
                            {static_cast<int64_t>(levels.size()) + 1}};
}

void HistogramOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("levels", levels);
}

namespace {

// Register the op so that we can add it to an Onnx model via the Builder
// for the purposes of testing
static OpDefinition
    histogramOpDef({OpDefinition::Inputs({{"data", supportedTypes}}),
                    OpDefinition::Outputs({{"counts", {outputType}}}),
                    OpDefinition::Attributes({{"levels", {"*"}},
                                              {"absoluteOfInput", {"*"}}})});

static OpCreator<HistogramOp> histogramOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Histogram, histogramOpDef}}),
    [](const OpCreatorInfo &info) {
      std::vector<float> levels =
          info.attributes.getAttribute<Attributes::Floats>("levels");
      bool absoluteOfInput =
          info.attributes.getAttribute<Attributes::Int>("absoluteOfInput");
      return std::unique_ptr<HistogramOp>(
          new HistogramOp(info.opid, levels, absoluteOfInput, info.settings));
    },
    true);

} // namespace

} // namespace popart
