#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
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
  return make_unique<DropoutOp>(*this);
}

void DropoutOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());

  if (output->n() != 1) {
    throw error("The op, \"{}\" has {} outputs. In Poponnx the Dropout op only "
                "supports a single output tensor. The optional 'mask' output "
                "is not currently supported.",
                str(),
                output->n());
  }
}

std::vector<std::unique_ptr<Op>> DropoutOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<DropoutGradOp>(*this));
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
  return make_unique<DropoutGradOp>(*this);
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

      return std::unique_ptr<Op>(new DropoutOp(_opid, ratio, settings));
    },
    true);

} // namespace
} // namespace poponnx
