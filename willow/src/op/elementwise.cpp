// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/broadcastutil.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/tensor.hpp>

namespace {
using namespace popart;

view::RegMap binaryFwdRegMapImpl(const ElementWiseBinaryBaseOp &op,
                                 InIndex argIndex) {
  auto out_shape = op.outShape(op.getOutIndex());
  auto in_shape  = op.inShape(argIndex);

  return [out_shape, in_shape](const view::Region &r) {
    auto out_size  = out_shape.size();
    auto arg_shape = padShape(in_shape, out_size, int64_t{1});
    auto lower     = padShape(r.getLower(), out_size, int64_t{0});
    auto upper     = padShape(r.getUpper(), out_size, int64_t{1});

    // broadcasting
    for (int i = 0; i < out_shape.size(); i++) {
      if (arg_shape[i] == 1 && out_shape[i] > 1) {
        upper[i] = out_shape[i];
      }
    }

    return view::Regions(1, view::Region{lower, upper});
  };
}

view::RegMap binaryBwdRegMapImpl(const ElementWiseBinaryBaseOp &op,
                                 InIndex argIndex) {
  auto arg_shape = op.inShape(argIndex);
  auto arg_size  = arg_shape.size();
  auto out_shape = unpadShape(op.outShape(op.getOutIndex()), arg_size);

  return [arg_size, out_shape, arg_shape](const view::Region &r) {
    auto lower = unpadShape(r.getLower(), arg_size);
    auto upper = unpadShape(r.getUpper(), arg_size);

    // unbroadcasting
    for (int i = 0; i < out_shape.size(); i++) {
      if (arg_shape[i] == 1 && out_shape[i] > 1) {
        lower[i] = 0;
        upper[i] = 1;
      }
    }

    return view::Regions(1, view::Region{lower, upper});
  };
}

} // namespace

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
       GradOpInType::GradOut},
      {getFwdArgInIndex(), ElementWiseUnaryOp::getInIndex(), GradOpInType::In}};

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

ElementWiseBinaryBaseOp::ElementWiseBinaryBaseOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void ElementWiseBinaryBaseOp::setup() {
  outInfo(getOutIndex()) =
      prettyNpOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()));
}

ElementWiseBinaryOp::ElementWiseBinaryOp(const OperatorIdentifier &_opid,
                                         const Op::Settings &_settings)
    : ElementWiseBinaryBaseOp(_opid, _settings) {}

std::vector<std::tuple<OperatorIdentifier, float>>
ElementWiseBinaryOp::inplacePriorityDefault() const {
  auto outSize  = outInfo(getOutIndex()).nelms();
  auto arg0Size = inInfo(getArg0InIndex()).nelms();
  auto arg1Size = inInfo(getArg1InIndex()).nelms();

  std::vector<std::tuple<OperatorIdentifier, float>> result;

  if (hasLhsInplaceVariant() && outSize == arg0Size) {
    auto lhsPriority = getInplacePriority(getLhsOperatorIdentifier());
    result.emplace_back(getLhsOperatorIdentifier(), lhsPriority);
  }
  if (hasRhsInplaceVariant() && outSize == arg1Size) {
    auto rhsPriority = getInplacePriority(getRhsOperatorIdentifier());
    result.emplace_back(getRhsOperatorIdentifier(), rhsPriority);
  }

  return result;
}

std::unique_ptr<Op> ElementWiseBinaryOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (hasLhsInplaceVariant() && operator_id == getLhsOperatorIdentifier()) {
    return getLhsInplaceVariant();
  } else if (hasRhsInplaceVariant() &&
             operator_id == getRhsOperatorIdentifier()) {
    return getRhsInplaceVariant();
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

view::RegMap ElementWiseBinaryOp::fwdRegMap(InIndex argIndex, OutIndex) const {
  return binaryFwdRegMapImpl(*this, argIndex);
}

view::RegMap ElementWiseBinaryOp::bwdRegMap(InIndex argIndex, OutIndex) const {
  return binaryBwdRegMapImpl(*this, argIndex);
}

void ElementWiseBinaryOp::setInplacePriority(
    const OperatorIdentifier &opidParam,
    float priority) {
  inplacePriorities[opidParam] = priority;
}

float ElementWiseBinaryOp::getInplacePriority(
    const OperatorIdentifier &opidParam) const {
  constexpr float defaultPriority = 10.0f;

  if (inplacePriorities.count(opidParam) == 0) {
    return defaultPriority;
  } else {
    return inplacePriorities.at(opidParam);
  }
}

bool ElementWiseBinaryOp::hasLhsInplaceVariant() const { return false; }

bool ElementWiseBinaryOp::hasRhsInplaceVariant() const { return false; }

std::unique_ptr<Op> ElementWiseBinaryOp::getLhsInplaceVariant() const {
  throw error("Operator {} cannot return LHS inplace variant", opid);
}

std::unique_ptr<Op> ElementWiseBinaryOp::getRhsInplaceVariant() const {
  throw error("Operator {} cannot return RHS inplace variant", opid);
}

OperatorIdentifier ElementWiseBinaryOp::getLhsOperatorIdentifier() const {
  throw error("Operator {} does not have LHS OperatorIdentifier", opid);
}

OperatorIdentifier ElementWiseBinaryOp::getRhsOperatorIdentifier() const {
  throw error("Operator {} does not have RHS OperatorIdentifier", opid);
}

ElementWiseBinaryInplaceLhsOp::ElementWiseBinaryInplaceLhsOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &_settings)
    : ElementWiseBinaryBaseOp(_opid, _settings) {}

view::Regions ElementWiseBinaryInplaceLhsOp::modifies(InIndex index) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    throw error("Invalid index passed to modifies method for Operator {}",
                opid);
  }
}

view::Regions ElementWiseBinaryInplaceLhsOp::aliases(InIndex index,
                                                     OutIndex) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    throw error("Invalid index passed to aliases method for Operator {}", opid);
  }
}

view::RegMap ElementWiseBinaryInplaceLhsOp::fwdRegMap(InIndex argIndex,
                                                      OutIndex) const {
  return binaryFwdRegMapImpl(*this, argIndex);
}

view::RegMap ElementWiseBinaryInplaceLhsOp::bwdRegMap(InIndex argIndex,
                                                      OutIndex) const {
  return binaryBwdRegMapImpl(*this, argIndex);
}

ElementWiseBinaryInplaceRhsOp::ElementWiseBinaryInplaceRhsOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &_settings)
    : ElementWiseBinaryBaseOp(_opid, _settings) {}

view::Regions ElementWiseBinaryInplaceRhsOp::modifies(InIndex index) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else {
    throw error("Invalid index passed to modifies method for Operator {}",
                opid);
  }
}

view::Regions ElementWiseBinaryInplaceRhsOp::aliases(InIndex index,
                                                     OutIndex) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else {
    throw error("Invalid index passed to aliases method for Operator {}", opid);
  }
}

view::RegMap ElementWiseBinaryInplaceRhsOp::fwdRegMap(InIndex argIndex,
                                                      OutIndex) const {
  return binaryFwdRegMapImpl(*this, argIndex);
}

view::RegMap ElementWiseBinaryInplaceRhsOp::bwdRegMap(InIndex argIndex,
                                                      OutIndex) const {
  return binaryBwdRegMapImpl(*this, argIndex);
}

BinaryComparisonOp::BinaryComparisonOp(const OperatorIdentifier &_opid,
                                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void BinaryComparisonOp::setup() {
  outInfo(getOutIndex()) = {DataType::BOOL,
                            prettyNpOut(inInfo(getArg0InIndex()).shape(),
                                        inInfo(getArg1InIndex()).shape())};
}

} // namespace popart
