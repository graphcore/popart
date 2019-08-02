#include <memory>
#include <popart/ir.hpp>
#include <popart/op/dropout.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

DropoutBaseOp::DropoutBaseOp(const OperatorIdentifier &opid_,
                             float ratio_,
                             uint32_t seedModifier_,
                             const Op::Settings &settings_)
    : Op(opid_, settings_), ratio(ratio_), seedModifier(seedModifier_) {}

uint32_t DropoutBaseOp::getSeedModifier() const { return seedModifier; }

void DropoutBaseOp::setSeedModifier(uint32_t sm) { seedModifier = sm; }

float DropoutBaseOp::getRatio() const { return ratio; }

void DropoutBaseOp::setRatio(float r) { ratio = r; }

float DropoutBaseOp::getSubgraphValue() const { return getLowSubgraphValue(); }

DropoutOp::DropoutOp(const OperatorIdentifier &_opid,
                     float ratio_,
                     const Op::Settings &settings_)
    : DropoutBaseOp(_opid,
                    ratio_,
                    settings_.getIr().getAndIncrementDropoutSeedModifier(),
                    settings_) {}

std::unique_ptr<Op> DropoutOp::clone() const {
  return std::make_unique<DropoutOp>(*this);
}

void DropoutOp::setup() {
  if (output->n() > 1) {
    output_mask                = true;
    outInfo(getOutIndex())     = inInfo(getInIndex());
    outInfo(getMaskOutIndex()) = {DataType::BOOL, inInfo(getInIndex()).shape()};
  } else {
    outInfo(getOutIndex()) = inInfo(getInIndex());
  }

  if (getIr().isTraining()) {
    auto tensor_id = fmt::format("Dropout({})_seed", id);
    createAndConnectOutTensor(getSeedOutIndex(), tensor_id);
    outInfo(getSeedOutIndex()) = {DataType::UINT32, {2}};
  }
}

std::vector<std::unique_ptr<Op>> DropoutOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<DropoutGradOp>(*this));
  return upops;
}

void DropoutOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
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
    : DropoutBaseOp(Onnx::GradOperators::DropoutGrad,
                    fwdOp.getRatio(),
                    fwdOp.getSeedModifier(),
                    fwdOp.getSettings()) {}

std::unique_ptr<Op> DropoutGradOp::clone() const {
  return std::make_unique<DropoutGradOp>(*this);
}

void DropoutGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradInIndex());
}

const std::vector<GradInOutMapper> &DropoutGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), DropoutOp::getOutIndex(), GradOpInType::GRADOUT},
      {getSeedInIndex(), DropoutOp::getSeedOutIndex(), GradOpInType::OUT}};
  return inInfo;
}

const std::map<int, int> &DropoutGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DropoutOp::getInIndex()}};
  return outInfo;
}

namespace {
static OpCreator<DropoutOp> dropoutOpCreator(
    {Onnx::Operators::Dropout_6,
     Onnx::Operators::Dropout_7,
     Onnx::Operators::Dropout_10},
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
