#include <poponnx/makeunique.hpp>
#include <poponnx/op/mean.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

MeanOp::MeanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> MeanOp::clone() const { return make_unique<MeanOp>(*this); }

std::vector<std::unique_ptr<Op>> MeanOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  for (int i = 0; i < input->n(); ++i) {
    upops.push_back(make_unique<MeanGradOp>(*this, i));
  }

  return upops;
}

void MeanOp::setup() {

  outInfo(getOutIndex()) = inInfo(0);

  if (opid.version == 6) {
    // In version 6 all inputs must be the same shape
    for (int i = 1; i < input->n(); ++i) {
      if (inInfo(i) != outInfo(getOutIndex())) {
        throw error("Inputs to {} do not all the same type & shape", opid);
      }
    }
  } else {
    // In version 8 inputs are broadcast
    for (int i = 1; i < input->n(); ++i) {
      outInfo(getOutIndex()) = npOut(outInfo(getOutIndex()), inInfo(i));
    }
  }
}

bool MeanOp::canBeReplacedByIdentity() { return (input->n() == 1); }

MeanGradOp::MeanGradOp(const MeanOp &op_, InIndex inputIndex)
    : Op(Onnx::GradOperators::MeanGrad, op_.getSettings()),
      fwdIndex(inputIndex), numFwdOpInputs(op_.input->n()) {

  gradOutToNonGradInInfo = {{getOutIndex(), fwdIndex}};

  gradInputInfoVec = {
      {getGradInIndex(), MeanOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdInIndex(), fwdIndex, GradOpInType::IN}};
}

const std::map<int, int> &MeanGradOp::gradOutToNonGradIn() const {
  return gradOutToNonGradInInfo;
}

const std::vector<GradInOutMapper> &MeanGradOp::gradInputInfo() const {
  return gradInputInfoVec;
}

void MeanGradOp::setup() { outInfo(getOutIndex()) = inInfo(getFwdInIndex()); }

namespace {
static OpCreator<MeanOp> opCreator({Onnx::Operators::Mean_6,
                                    Onnx::Operators::Mean_8});
} // namespace

} // namespace poponnx
