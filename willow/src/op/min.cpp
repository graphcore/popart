#include <poponnx/makeunique.hpp>
#include <poponnx/op/min.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

MinOp::MinOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> MinOp::clone() const { return make_unique<MinOp>(*this); }

std::vector<std::unique_ptr<Op>> MinOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  for (int i = 0; i < input->n(); ++i) {
    upops.push_back(make_unique<MinGradOp>(*this, i));
  }
  return upops;
}

void MinOp::setup() {

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

bool MinOp::canBeReplacedByIdentity() { return (input->n() == 1); }

MinGradOp::MinGradOp(const MinOp &op_, InIndex inputIndex)
    : Op(Onnx::GradOperators::MinGrad, op_.getSettings()),
      fwdIndex(inputIndex) {
  gradOutToNonGradInInfo = {{getOutIndex(), fwdIndex}};

  gradInputInfoVec = {
      {getGradInIndex(), MinOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdInIndex(), fwdIndex, GradOpInType::IN},
      {getFwdOutInIndex(), MinOp::getOutIndex(), GradOpInType::OUT}};
}

const std::map<int, int> &MinGradOp::gradOutToNonGradIn() const {
  return gradOutToNonGradInInfo;
}

const std::vector<GradInOutMapper> &MinGradOp::gradInputInfo() const {
  return gradInputInfoVec;
}

void MinGradOp::setup() { outInfo(getOutIndex()) = inInfo(getFwdInIndex()); }

namespace {
static OpCreator<MinOp> addOpCreator({Onnx::Operators::Min_6,
                                      Onnx::Operators::Min_8});
} // namespace

} // namespace poponnx
