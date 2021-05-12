// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op.hpp>
#include <popart/op/ctcbeamsearch.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {
CtcBeamSearchDecoderOp::CtcBeamSearchDecoderOp(
    const popart::OperatorIdentifier &_opid,
    unsigned _blankClass,
    unsigned _beamWidth,
    unsigned _topPaths,
    const popart::Op::Settings &settings_)
    : Op(_opid, settings_), blankClass(_blankClass), beamWidth(_beamWidth),
      topPaths(_topPaths) {}

std::unique_ptr<Op> CtcBeamSearchDecoderOp::clone() const {
  return std::make_unique<CtcBeamSearchDecoderOp>(*this);
}

void CtcBeamSearchDecoderOp::setup() {
  // Validate logProbs input.
  const auto &logProbsInInfo  = inInfo(getLogProbsInIndex());
  const auto &logProbsInShape = inShape(getLogProbsInIndex());

  if (logProbsInShape.size() != 3) { // Is shape correct?
    throw error(
        "Unsupported shape for input {} of Op {} (expected a 'logarithmized "
        "probabilities' tensor of rank 3, got a tensor with shape {}).",
        getLogProbsInIndex(),
        str(),
        logProbsInShape);
  }

  if (logProbsInInfo.getDataTypeInfo()->isFixedPoint()) { // Is dtype correct?
    throw error(
        "Unsupported data type for input {} of Op {} (expected a "
        "'logarithmized "
        "probabilities' tensor comprised of floating point data, got {}).",
        getLogProbsInIndex(),
        str(),
        logProbsInInfo.getDataTypeInfo()->type());
  }

  // Set attributes.
  maxTime    = logProbsInShape.at(0);
  batchSize  = logProbsInShape.at(1);
  numClasses = logProbsInShape.at(2);

  // Validate dataLengths input.
  const auto &dataLengthsInInfo  = inInfo(getDataLengthsInIndex());
  const auto &dataLengthsInShape = inShape(getDataLengthsInIndex());

  Shape expectedDataLengthsShape = {batchSize};

  if (dataLengthsInShape != expectedDataLengthsShape) { // Is shape correct?
    throw error(
        "Unsupported shape for input {} of Op {} (expected a 'data lengths' "
        "tensor of shape {}, got {}).",
        getDataLengthsInIndex(),
        str(),
        expectedDataLengthsShape,
        dataLengthsInShape);
  }

  if (dataLengthsInInfo.getDataTypeInfo()->type() !=
      DataType::UINT32) { // Is dtype correct?
    throw error("Unsupported data type for input {} of Op {} (expected a "
                "'input lengths' tensor of type {}, got {}).",
                getDataLengthsInIndex(),
                str(),
                DataType::UINT32,
                dataLengthsInInfo.getDataTypeInfo()->type());
  }

  // Set out info.
  outInfo(getLabelProbsOutIndex())
      .set(logProbsInInfo.dataType(), {getBatchSize(), getTopPaths()});
  outInfo(getLabelLengthsOutIndex())
      .set(DataType::UINT32, {getBatchSize(), getTopPaths()});
  outInfo(getDecodedLabelsOutIndex())
      .set(DataType::UINT32, {getBatchSize(), getTopPaths(), getMaxTime()});
}

void CtcBeamSearchDecoderOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("blank", getBlankClass());
  os.appendAttribute("beamWidth", getBeamWidth());
  os.appendAttribute("topPaths", getTopPaths());
}

void CtcBeamSearchDecoderOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("blank", getBlankClass());
  os.appendAttribute("beamWidth", getBeamWidth());
  os.appendAttribute("topPaths", getTopPaths());
}

std::vector<std::unique_ptr<Op>> CtcBeamSearchDecoderOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  return upops;
}

float CtcBeamSearchDecoderOp::getSubgraphValue() const {
  return getHighSubgraphValue();
}

bool CtcBeamSearchDecoderOp::requiresRandomSeed() const { return false; }

// Op creator.
namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T1 = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T2 = {DataType::UINT32};

static OpDefinition ctcBeamSearchDecoderOpDef({
    OpDefinition::Inputs({
        {"logProbs", T1},
        {"dataLengths", T2},
    }),
    OpDefinition::Outputs({
        {"labelProbs", T1},
        {"labelLengths", T2},
        {"decodedLabels", T2},
    }),
    OpDefinition::Attributes({
        {"blankClass", {"*"}},
        {"beamWidth", {"*"}},
        {"topPaths", {"*"}},
    }),
});

static OpCreator<CtcBeamSearchDecoderOp> ctcBeamSearchDecoderOpCreator(
    OpDefinitions({{
        Onnx::CustomOperators::CtcBeamSearchDecoder,
        ctcBeamSearchDecoderOpDef,
    }}),
    [](const OpCreatorInfo &info) {
      unsigned blankClass =
          info.attributes.getAttribute<Attributes::Int>("blank", 0);
      unsigned beamWidth =
          info.attributes.getAttribute<Attributes::Int>("beamWidth", 100);
      unsigned topPaths =
          info.attributes.getAttribute<Attributes::Int>("topPaths", 1);

      return std::unique_ptr<CtcBeamSearchDecoderOp>(new CtcBeamSearchDecoderOp(
          info.opid, blankClass, beamWidth, topPaths, info.settings));
    },
    true);
} // namespace
} // namespace popart