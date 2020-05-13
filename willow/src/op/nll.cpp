// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <sstream>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/nll.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::unique_ptr<Op> NllOp::clone() const {
  return std::make_unique<NllOp>(*this);
}

std::vector<std::unique_ptr<Op>> NllOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<NllGradOp>(*this));
  return upops;
}

void NllOp::setup() {

  const auto &labelsInInfo = inInfo(getLabelInIndex());
  if (!labelsInInfo.getDataTypeInfo()->isFixedPoint()) {
    throw error(
        "Expected the label tensor NllOp to be fixed point, not the case "
        "for input with info: {}. This error for Op {}. ",
        labelsInInfo,
        str());
  }

  const auto &probsInInfo = inInfo(getProbsInIndex());
  const auto &labelInInfo = inInfo(getLabelInIndex());
  // Outputs a loss for each label index.
  // Same shape as label input, same datatype as probs input
  outInfo(getOutIndex()).set(probsInInfo.dataType(), labelInInfo.shape());
}

NllOp::NllOp(const OperatorIdentifier &_opid,
             const boost::optional<int> ignoreIndex,
             const ReductionType reduction,
             const Op::Settings &settings)
    : LossOp(_opid, settings), reduction_(reduction),
      ignoreIndex_(ignoreIndex) {}

void NllOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduction_type", static_cast<int64_t>(reduction_));
  if (hasIgnoreIndex()) {
    os.appendAttribute("ignore_index", static_cast<int64_t>(*ignoreIndex_));
  }
}

int NllOp::getIgnoreIndex() const {
  if (hasIgnoreIndex()) {
    return ignoreIndex_.get();
  } else {
    throw error("Cannot getIgnoreIndex for {}, as it has none", str());
  }
}

void NllGradOp::setup() {

  // connect the loss scaling tensor if is non-const
  if (!getIr().getOptimizer().lossScaling().isConst()) {
    connectInTensor(NllGradOp::getLossScalingInIndex(),
                    getIr().getOptimizer().getLossScalingTensorId(
                        inInfo(getProbsInIndex()).dataType()));
  }

  // gradient of probs has same shape as probs
  outInfo(getOutIndex()) = inInfo(getProbsInIndex());
}

NllGradOp::NllGradOp(const NllOp &op_)
    : Op(Onnx::CustomGradOperators::NllGrad, op_.getSettings()),
      lossId_(op_.outId(NllOp::getOutIndex())),
      reduction_(op_.getReductionType()),
      ignoreIndex_(op_.getOptionalIgnoreIndex()) {}

std::unique_ptr<Op> NllGradOp::clone() const {
  return std::make_unique<NllGradOp>(*this);
}

const std::vector<GradInOutMapper> &NllGradOp::gradInputInfo() const {
  // input at index 0 : labelIn()
  // input at index 1 : probsIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {getLabelInIndex(), NllOp::getLabelInIndex(), GradOpInType::In},
      {getProbsInIndex(), NllOp::getProbsInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &NllGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index probsIn()
  // the op ONLY computes the gradient of probs,
  // no gradient for label (one could interpret the
  // int as a sparse vector, but not neat)
  static const std::map<int, int> outInfo = {
      {getOutIndex(), NllOp::getProbsInIndex()}};
  return outInfo;
}

void NllGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduction_type", static_cast<int64_t>(reduction_));
  if (hasIgnoreIndex()) {
    os.appendAttribute("ignore_index", static_cast<int64_t>(*ignoreIndex_));
  }
}

int NllGradOp::getIgnoreIndex() const {
  if (hasIgnoreIndex()) {
    return ignoreIndex_.get();
  } else {
    throw error("Cannot getIgnoreIndex for {}, as it has none", str());
  }
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition::DataTypes T2 = {DataType::INT32, DataType::UINT32};

static OpDefinition
    nlllossOpDef({OpDefinition::Inputs({{"A", T1}, {"B", T2}}),
                  OpDefinition::Outputs({{"C", T1}}),
                  OpDefinition::Attributes({{"reduction", {"*"}},
                                            {"ignore_index", {"*"}}})});

static OpCreator<NllOp> nlllossOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Nll, nlllossOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr = {}) -> std::unique_ptr<Op> {
      std::string reductionStr =
          attr.getAttribute<Attributes::String>("reduction");
      ReductionType reduction = LossOp::reductionTypeFromString(reductionStr);

      boost::optional<int> ignoreIndex;
      if (attr.hasAttribute("ignoreIndex")) {
        ignoreIndex = attr.getAttribute<Attributes::Int>("ignoreIndex");
      }
      return std::unique_ptr<NllOp>(
          new NllOp(_opid, ignoreIndex, reduction, settings));
    },
    true);

} // namespace

} // namespace popart
