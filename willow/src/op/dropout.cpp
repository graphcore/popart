// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dropout.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

DropoutOp::DropoutOp(const OperatorIdentifier &opid_,
                     float ratio_,
                     RandomReferenceId referenceId_,
                     bool outputMask_,
                     const Op::Settings &settings_)
    : DropoutBaseOp(opid_, ratio_, settings_), referenceId(referenceId_),
      outputMask(outputMask_) {}

DropoutOp::DropoutOp(const OperatorIdentifier &_opid,
                     float ratio_,
                     const Op::Settings &settings_)
    : DropoutBaseOp(_opid, ratio_, settings_) {
  referenceId = getIr().getAndIncrementRandomReferenceId();
}

std::unique_ptr<Op> DropoutOp::clone() const {
  return std::make_unique<DropoutOp>(*this);
}

void DropoutOp::setup() {
  if (output->n() > 1) {
    setOutputMask(true);
    outInfo(getOutIndex())     = inInfo(getInIndex());
    outInfo(getMaskOutIndex()) = {DataType::BOOL, inInfo(getInIndex()).shape()};
  } else {
    outInfo(getOutIndex()) = inInfo(getInIndex());
  }
}

std::vector<std::unique_ptr<Op>> DropoutOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<DropoutGradOp>(*this));
  return upops;
}

void DropoutOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("randomReferenceId", referenceId);
}

void DropoutOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("ratio", getRatio());
}

// Get the reference TensorId used for poplibs call for mask generation.
// Note that the reference Tensor cannot just be the Tensor to be masked, as the
// mask generated depends on the layout of the input and not just the random
// seed. It is required that forwards, recompute and backwards masks are all the
// same, so the same reference tensor must be used in for these cases
TensorId DropoutOp::getReferenceTensorId() {
  auto name = str();

  auto refTensorId =
      getIr().getOrSetRandomReferenceTensor(referenceId, inId(getInIndex()));

  logging::debug(
      "Op {}: using tensor [{}] to use as mask reference.", name, refTensorId);

  return refTensorId;
}

DropoutGradOp::DropoutGradOp(const DropoutOp &fwdOp)
    : DropoutOp(fwdOp.opid,
                fwdOp.getRatio(),
                fwdOp.getReferenceId(),
                fwdOp.getOutputMask(),
                fwdOp.getSettings()) {}

std::unique_ptr<Op> DropoutGradOp::clone() const {
  return std::make_unique<DropoutGradOp>(*this);
}

const std::vector<GradInOutMapper> &DropoutGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), DropoutOp::getOutIndex(), GradOpInType::GradOut},
      // Dropout and DropoutGrad inherit from the same base op, so share the
      // same seed InIndex
      {getSeedInIndex(), DropoutOp::getSeedInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &DropoutGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DropoutOp::getInIndex()}};
  return outInfo;
}

namespace {

static OpDefinition::DataTypes T  = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::BOOL};

static OpDefinition
    dropoutOpDef({OpDefinition::Inputs({{"data", T}}),
                  OpDefinition::Outputs({{"output", T}, {"mask", T1}}),
                  OpDefinition::Attributes({{"ratio", {"*"}}})});

static OpCreator<DropoutOp> dropoutOpCreator(
    OpDefinitions({{Onnx::Operators::Dropout_6, dropoutOpDef},
                   {Onnx::Operators::Dropout_7, dropoutOpDef},
                   {Onnx::Operators::Dropout_10, dropoutOpDef}}),
    [](const OpCreatorInfo &info) {
      float ratio = DropoutBaseOp::validateRatioAttribute(info);
      return std::unique_ptr<Op>(
          new DropoutOp(info.opid, ratio, info.settings));
    },
    true);

} // namespace
} // namespace popart
