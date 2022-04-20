// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <string>
#include <vector>
#include <popart/error.hpp>
#include <popart/op/ctc.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/loss.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
struct OperatorIdentifier;
} // namespace popart

static size_t batchSizeDimension       = 1;
static size_t maxInputLengthDimension  = 0;
static size_t maxTargetLengthDimension = 1;
static size_t numClassesDimension      = 2;

namespace popart {

CtcOp::CtcOp(const OperatorIdentifier &_opid,
             const ReductionType reduction_,
             const unsigned blank_,
             const Op::Settings &_settings,
             const DataType userOutputType_)
    : LossOp(_opid, _settings, reduction_), blank(blank_),
      userOutputType(userOutputType_) {}

std::unique_ptr<Op> CtcOp::clone() const {
  return std::make_unique<CtcOp>(*this);
}

std::vector<std::unique_ptr<Op>> CtcOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<CtcGradOp>(*this));
  return upops;
}

unsigned CtcOp::getBatchSize() const {
  return inShape(getLogProbsInIndex()).at(batchSizeDimension);
}

unsigned CtcOp::getMaxInputLength() const {
  return inShape(getLogProbsInIndex()).at(maxInputLengthDimension);
}

unsigned CtcOp::getMaxTargetLength() const {
  return inShape(getTargetsInIndex()).at(maxTargetLengthDimension);
}

unsigned CtcOp::getNumClasses() const {
  return inShape(getLogProbsInIndex()).at(numClassesDimension);
}

void CtcOp::setup() {

  // Check first input.
  const auto &logProbsInInfo  = inInfo(getLogProbsInIndex());
  const auto &logProbsInShape = inShape(getLogProbsInIndex());

  if (logProbsInShape.size() != 3) {
    throw error(
        "Unsupported shape for input {} of Op {} (expected a 'logarithmized "
        "probabilities' tensor of rank 3, got a tensor with shape {}).",
        getLogProbsInIndex(),
        str(),
        logProbsInShape);
  }

  if (logProbsInInfo.getDataTypeInfo()->isFixedPoint()) {
    throw error(
        "Unsupported data type for input {} of Op {} (expected a "
        "'logarithmized "
        "probabilities' tensor comprised of floating point data, got {}).",
        getLogProbsInIndex(),
        str(),
        logProbsInInfo.getDataTypeInfo()->type());
  }

  if (userOutputType != DataType::UNDEFINED &&
      getDataTypeInfoMap().at(userOutputType).isFixedPoint()) {
    throw error(
        "Unsupported data type for output {} of Op {} (expected a floating "
        "point type, got {}).",
        getCtcLossOutIndex(),
        str(),
        userOutputType);
  }

  // Check second input.
  const auto &targetsInInfo  = inInfo(getTargetsInIndex());
  const auto &targetsInShape = inShape(getTargetsInIndex());

  auto batchSize = getBatchSize();
  if (targetsInShape.size() != 2 || targetsInShape.at(0) != batchSize) {
    throw error(
        "Unsupported shape for input {} of Op {} (expected a 'targets' tensor "
        "of rank 2 with the size of the first dimension being {} to match the "
        "size of the second dimension of the 'logarithmized probabilities' "
        "tensor, got {}).",
        getTargetsInIndex(),
        str(),
        batchSize,
        logProbsInShape);
  }

  if (targetsInInfo.getDataTypeInfo()->type() != DataType::UINT32) {
    throw error(
        "Unsupported data type for input {} of Op {} (expected a 'targets' "
        "tensor of type {}, got {}).",
        getLogProbsInIndex(),
        str(),
        DataType::UINT32,
        targetsInInfo.getDataTypeInfo()->type());
  }

  // Check third input.
  Shape expectedLengthShape;
  expectedLengthShape.emplace_back(batchSize);

  const auto &inputLengthsInInfo  = inInfo(getInputLengthsInIndex());
  const auto &inputLengthsInShape = inShape(getInputLengthsInIndex());

  if (inputLengthsInShape != expectedLengthShape) {
    throw error(
        "Unsupported shape for input {} of Op {} (expected a 'input lengths' "
        "tensor of shape {}, got {}).",
        getInputLengthsInIndex(),
        str(),
        expectedLengthShape,
        inputLengthsInShape);
  }

  if (inputLengthsInInfo.getDataTypeInfo()->type() != DataType::UINT32) {
    throw error("Unsupported data type for input {} of Op {} (expected a "
                "'input lengths' tensor of type {}, got {}).",
                getInputLengthsInIndex(),
                str(),
                DataType::UINT32,
                inputLengthsInInfo.getDataTypeInfo()->type());
  }

  // Check fourth input.
  const auto &targetLengthsInInfo  = inInfo(getTargetLengthsInIndex());
  const auto &targetLengthsInShape = inShape(getTargetLengthsInIndex());

  if (targetLengthsInShape != expectedLengthShape) {
    throw error(
        "Unsupported shape for input {} of Op {} (expected a 'target lengths' "
        "tensor of shape {}, got {}).",
        getTargetLengthsInIndex(),
        str(),
        expectedLengthShape,
        targetLengthsInShape);
  }

  if (targetLengthsInInfo.getDataTypeInfo()->type() != DataType::UINT32) {
    throw error("Unsupported data type for input {} of Op {} (expected a "
                "'target lengths' tensor of type {}, got {}).",
                getTargetLengthsInIndex(),
                str(),
                DataType::UINT32,
                targetLengthsInInfo.getDataTypeInfo()->type());
  }

  // Default output data types to the type of the first input.
  DataType outputType = logProbsInInfo.dataType();

  // User can specify output type.
  if (userOutputType != DataType::UNDEFINED) {
    outputType = userOutputType;
  }

  // Loss output info.
  if (getReductionType() != ReductionType::NoReduction) {
    // With any reduction, output is scalar.
    outInfo(getCtcLossOutIndex()).set(outputType, {});
  } else {
    // With no reduction, output is [N].
    outInfo(getCtcLossOutIndex()).set(outputType, {batchSize});
  }

  // With no reduction, output is [T, N, C].
  outInfo(getLogProbsGradientWrtCtcLossOutIndex())
      .set(outputType, {getMaxInputLength(), batchSize, getNumClasses()});
}

void CtcOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduction_type",
                     static_cast<int64_t>(getReductionType()));
  os.appendAttribute("blank", static_cast<int64_t>(blank));
}

void CtcGradOp::setup() {
  // gradient of probs has same shape as probs
  outInfo(getLogProbsGradientOutIndex()) = logProbsInfo;
}

CtcGradOp::CtcGradOp(const CtcOp &op_)
    : Op(Onnx::CustomGradOperators::CtcGrad, op_.getSettings()),
      reduction(op_.getReductionType()),
      logProbsInfo(op_.inInfo(CtcOp::getLogProbsInIndex())) {}

std::unique_ptr<Op> CtcGradOp::clone() const {
  return std::make_unique<CtcGradOp>(*this);
}

const std::vector<GradInOutMapper> &CtcGradOp::gradInputInfo() const {
  // input at index 0 : the gradient output of the fwd op
  // input at index 1 : the gradient of the loss produced by the fwd op
  static const std::vector<GradInOutMapper> inInfo = {
      {getLogProbsGradientWrtCtcLossInIndex(),
       CtcOp::getLogProbsGradientWrtCtcLossOutIndex(),
       GradOpInType::Out},
      {getTargetLengthsInIndex(),
       CtcOp::getTargetLengthsInIndex(),
       GradOpInType::In},
      {getCtcLossGradientInIndex(),
       CtcOp::getCtcLossOutIndex(),
       GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &CtcGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's CTC op's probs tensor,
  // no gradient for other tensors
  static const std::map<int, int> outInfo = {
      {getLogProbsGradientOutIndex(), CtcOp::getLogProbsInIndex()}};
  return outInfo;
}

void CtcGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduction_type", static_cast<int64_t>(reduction));
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T2 = {DataType::UINT32};

static OpDefinition ctclossOpDef(
    {OpDefinition::Inputs({{"A", T1}, {"B", T2}, {"C", T2}, {"D", T2}}),
     OpDefinition::Outputs({{"E", T1}, {"F", T1}}),
     OpDefinition::Attributes({{"reduction", {"*"}},
                               {"blank", {"*"}},
                               {"outDataType", {"FLOAT|FLOAT16|UNDEFINED"}}})});

static OpCreator<CtcOp> ctclossOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Ctc, ctclossOpDef}}),
    [](const OpCreatorInfo &info) {
      std::string reductionStr =
          info.attributes.getAttribute<Attributes::String>("reduction");

      unsigned blank = static_cast<unsigned>(
          info.attributes.getAttribute<Attributes::Int>("blank"));
      ReductionType reduction = LossOp::reductionTypeFromString(reductionStr);

      // Get data type from attributes.
      int64_t i64_to;
      info.attributes.set(i64_to, "outDataType");
      auto tpdt_to = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(i64_to);
      DataType outDataType = onnxutil::getDataType(tpdt_to);

      return std::unique_ptr<CtcOp>(
          new CtcOp(info.opid, reduction, blank, info.settings, outDataType));
    },
    true);

} // namespace

} // namespace popart
