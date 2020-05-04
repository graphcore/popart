// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/dropout.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

DropoutOp::DropoutOp(const OperatorIdentifier &opid_,
                     float ratio_,
                     uint32_t seedModifier_,
                     bool outputMask_,
                     const Op::Settings &settings_)
    : Op(opid_, settings_), ratio(ratio_), seedModifier(seedModifier_),
      output_mask(outputMask_) {}

uint32_t DropoutOp::getSeedModifier() const { return seedModifier; }

void DropoutOp::setSeedModifier(uint32_t sm) { seedModifier = sm; }

float DropoutOp::getRatio() const { return ratio; }

void DropoutOp::setRatio(float r) { ratio = r; }

float DropoutOp::getSubgraphValue() const { return getLowSubgraphValue(); }

DropoutOp::DropoutOp(const OperatorIdentifier &_opid,
                     float ratio_,
                     const Op::Settings &settings_)
    : DropoutOp(_opid,
                ratio_,
                settings_.getIr().getAndIncrementDropoutSeedModifier(),
                false,
                settings_) {}

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

void DropoutOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("ratio", ratio);

  // Appending the seedModfier ensures that caching can only occur
  // between dropout ops in the same layer, i.e. between the forward op,
  // the corresponding backwards op (if replaced by the dropout pattern),
  // and the corresponding recompute op (if the fwd op is cloned for
  // recomputation)
  os.appendAttribute("seedModifier", getSeedModifier());
}

// Dropout in testing mode can be replaced by the identity
bool DropoutOp::canBeReplacedByIdentity() {
  return (getIr().isTesting() || getIr().isEvaluation());
}

DropoutGradOp::DropoutGradOp(const DropoutOp &fwdOp)
    : DropoutOp(fwdOp.opid,
                fwdOp.getRatio(),
                fwdOp.getSeedModifier(),
                fwdOp.getOutputMask(),
                fwdOp.getSettings()) {}

std::unique_ptr<Op> DropoutGradOp::clone() const {
  return std::make_unique<DropoutGradOp>(*this);
}

const std::vector<GradInOutMapper> &DropoutGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), DropoutOp::getOutIndex(), GradOpInType::GradOut},
      // Dropout and DropoutGrad inheret from the same base op, so share the
      // same seed InIndex
      {getSeedInIndex(), getSeedInIndex(), GradOpInType::In}};
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
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      float ratio = attr.getAttribute<Attributes::Float>("ratio", 0.5f);
      // If invalid probability for ratio supplied, throw error.
      if (ratio <= float(0.) || ratio >= float(1.)) {
        throw error("{} ratio value {} is not valid. Please use a value in the "
                    "interval (0,1)",
                    _opid,
                    ratio);
      }
      return std::unique_ptr<Op>(new DropoutOp(_opid, ratio, settings));
    },
    true);

} // namespace
} // namespace popart
