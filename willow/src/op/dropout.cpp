#include <memory>
#include <poponnx/ir.hpp>
#include <poponnx/op/dropout.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

DropoutOp::DropoutOp(const OperatorIdentifier &_opid,
                     float ratio_,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), ratio(ratio_) {
  seedModifier = getIr().getAndIncrementDropoutSeedModifier();
}

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
  return (getIr().isTesting() || getIr().isEvaulation());
}

DropoutGradOp::DropoutGradOp(const DropoutOp &fwdOp)
    : Op(Onnx::GradOperators::DropoutGrad, fwdOp.getSettings()),
      ratio(fwdOp.getRatio()), seedModifier(fwdOp.getSeedModifier()) {}

std::unique_ptr<Op> DropoutGradOp::clone() const {
  return std::make_unique<DropoutGradOp>(*this);
}

void DropoutGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradInIndex());
}

const std::vector<GradInOutMapper> &DropoutGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      // Design note : seed tensor input to dropout op/gradop not in Ir
      {getGradInIndex(), DropoutOp::getOutIndex(), GradOpInType::GRADOUT}};
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
} // namespace poponnx
