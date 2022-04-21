// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/variadic.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

VariadicOp::VariadicOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::vector<std::unique_ptr<Op>> VariadicOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  for (int i = 0; i < input->n(); ++i) {
    upops.push_back(getIthGrad(i));
  }
  return upops;
}

void VariadicOp::setup() {

  if (opid.version == 6) {
    // In version 6 all inputs must be the same shape
    for (int i = 1; i < input->n(); ++i) {
      if (inInfo(i) != inInfo(0)) {
        throw error("Inputs to {} do not all the same type & shape, "
                    "with opid.version = 6",
                    opid);
      }
    }
  }

  // set output info
  outInfo(getOutIndex()) = inInfo(0);
  // In version 8 inputs are broadcast
  for (int i = 1; i < input->n(); ++i) {
    outInfo(getOutIndex()) = prettyNpOut(outInfo(getOutIndex()), inInfo(i));
  }
}

bool VariadicOp::canBeReplacedByIdentity() const { return (input->n() == 1); }

VariadicGradOp::VariadicGradOp(const OperatorIdentifier &_opid,
                               const VariadicOp &op_,
                               InIndex inputIndex)
    : Op(_opid, op_.getSettings()), fwdIndex(inputIndex),
      fwdInputInfo(op_.inInfo(inputIndex)) {

  gradOutToNonGradInInfo = {{getOutIndex(), getFwdIndex()}};
}

std::unique_ptr<Op> VariadicGradOp::clone() const {
  return std::make_unique<VariadicGradOp>(*this);
}

LinearVariadicGradOp::LinearVariadicGradOp(const OperatorIdentifier &_opid,
                                           const VariadicOp &op_,
                                           InIndex index)
    : VariadicGradOp(_opid, op_, index) {}

std::unique_ptr<Op> LinearVariadicGradOp::clone() const {
  return std::make_unique<LinearVariadicGradOp>(*this);
}

NonLinearVariadicGradOp::NonLinearVariadicGradOp(
    const OperatorIdentifier &_opid,
    const VariadicOp &op_,
    InIndex inputIndex)
    : VariadicGradOp(_opid, op_, inputIndex) {

  gradInputInfoVec = {
      {getGradInIndex(), VariadicOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdInIndex(), getFwdIndex(), GradOpInType::In},
      {getFwdOutInIndex(), VariadicOp::getOutIndex(), GradOpInType::Out}};
}

std::unique_ptr<Op> NonLinearVariadicGradOp::clone() const {
  return std::make_unique<NonLinearVariadicGradOp>(*this);
}

const std::map<int, int> &VariadicGradOp::gradOutToNonGradIn() const {
  return gradOutToNonGradInInfo;
}

const std::vector<GradInOutMapper> &
NonLinearVariadicGradOp::gradInputInfo() const {
  return gradInputInfoVec;
}

void VariadicGradOp::setup() { outInfo(getOutIndex()) = fwdInputInfo; }

namespace {} // namespace

} // namespace popart
