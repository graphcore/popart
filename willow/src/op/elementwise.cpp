// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/op/elementwise.hpp>
#include <popart/tensor.hpp>

namespace popart {

ElementWiseUnaryOp::ElementWiseUnaryOp(const OperatorIdentifier &_opid,
                                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void ElementWiseUnaryOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

ElementWiseUnaryBooleanOp::ElementWiseUnaryBooleanOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void ElementWiseUnaryBooleanOp::setup() {
  outInfo(getOutIndex()) = {DataType::BOOL, inInfo(getInIndex()).shape()};
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

ElementWiseBinaryOp::ElementWiseBinaryOp(const OperatorIdentifier &_opid,
                                         const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void ElementWiseBinaryOp::setup() {
  outInfo(getOutIndex()) =
      npOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()));
}

BinaryComparisonOp::BinaryComparisonOp(const OperatorIdentifier &_opid,
                                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void BinaryComparisonOp::setup() {
  outInfo(getOutIndex()) = {DataType::BOOL,
                            npOut(inInfo(getArg0InIndex()).shape(),
                                  inInfo(getArg1InIndex()).shape())};
}

} // namespace popart
