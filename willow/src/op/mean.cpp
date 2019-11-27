#include <memory>
#include <popart/op/mean.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

MeanOp::MeanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : VariadicOp(_opid, settings_) {}

std::unique_ptr<Op> MeanOp::clone() const {
  return std::make_unique<MeanOp>(*this);
}

std::unique_ptr<Op> MeanOp::getIthGrad(int i) const {
  return std::make_unique<MeanArgGradOp>(*this, i);
}

MeanArgGradOp::MeanArgGradOp(const MeanOp &op_, InIndex inputIndex)
    : LinearVariadicGradOp(Onnx::GradOperators::MeanArgGrad, op_, inputIndex) {

  gradInputInfoVec = {
      {getGradInIndex(), VariadicOp::getOutIndex(), GradOpInType::GRADOUT}};

  nInputs = op_.input->n();
}

std::unique_ptr<Op> MeanArgGradOp::clone() const {
  return std::make_unique<MeanArgGradOp>(*this);
}

const std::vector<GradInOutMapper> &MeanArgGradOp::gradInputInfo() const {
  return gradInputInfoVec;
}

void MeanArgGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  LinearVariadicGradOp::appendOutlineAttributes(os);
  os.appendAttribute("scale", getScale());
}

namespace {
static OpCreator<MeanOp> opCreator({Onnx::Operators::Mean_6,
                                    Onnx::Operators::Mean_8});
} // namespace

} // namespace popart
