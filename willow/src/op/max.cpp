#include <poponnx/makeunique.hpp>
#include <poponnx/op/max.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

MaxOp::MaxOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> MaxOp::clone() const { return make_unique<MaxOp>(*this); }

std::vector<std::unique_ptr<Op>> MaxOp::getGradOps() {

  std::vector<std::unique_ptr<Op>> upops;
  for (int i = 0; i < input->n(); ++i) {
    upops.push_back(make_unique<MaxGradOp>(*this, i));
  }

  return upops;
}

void MaxOp::setup() {

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

bool MaxOp::canBeReplacedByIdentity() { return (input->n() == 1); }

MaxGradOp::MaxGradOp(const MaxOp &op_, InIndex inputIndex)
    : Op(Onnx::GradOperators::MaxGrad, op_.getSettings()),
      fwdIndex(inputIndex) {
  gradOutToNonGradInInfo = {{getOutIndex(), fwdIndex}};

  gradInputInfoVec = {
      {getGradInIndex(), MaxOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdInIndex(), fwdIndex, GradOpInType::IN},
      {getFwdOutInIndex(), MaxOp::getOutIndex(), GradOpInType::OUT}};
}

const std::map<int, int> &MaxGradOp::gradOutToNonGradIn() const {
  return gradOutToNonGradInInfo;
}

const std::vector<GradInOutMapper> &MaxGradOp::gradInputInfo() const {
  return gradInputInfoVec;
}

void MaxGradOp::setup() { outInfo(getOutIndex()) = inInfo(getFwdInIndex()); }

namespace {
static OpCreator<MaxOp> addOpCreator({Onnx::Operators::Max_6,
                                      Onnx::Operators::Max_8});
} // namespace

} // namespace poponnx
