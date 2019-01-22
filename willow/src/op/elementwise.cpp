#include <poponnx/op/elementwise.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ElementWiseUnaryOp::ElementWiseUnaryOp(const OperatorIdentifier &_opid,
                                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void ElementWiseUnaryOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

ElementWiseNonLinearUnaryGradOp::ElementWiseNonLinearUnaryGradOp(
    const OperatorIdentifier &_opid,
    const ElementWiseUnaryOp &op)
    : Op(_opid, op.getSettings()) {}

const std::vector<GradInOutMapper> &
ElementWiseNonLinearUnaryGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(),
       ElementWiseUnaryOp::getOutIndex(),
       GradOpInType::GRADOUT},
      {getFwdArgInIndex(), ElementWiseUnaryOp::getInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &
ElementWiseNonLinearUnaryGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ElementWiseUnaryOp::getInIndex()}};

  return outInfo;
}

void ElementWiseNonLinearUnaryGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getFwdArgInIndex());
}

} // namespace poponnx
