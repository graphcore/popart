// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <memory>
#include <popart/op/groupnorm.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

GroupNormOp::GroupNormOp(const OperatorIdentifier &opid_,
                         int64_t num_groups_,
                         float epsilon_,
                         const Op::Settings &settings_)
    : Op(opid_, settings_), num_groups(num_groups_), epsilon(epsilon_) {}

std::unique_ptr<Op> GroupNormOp::clone() const {
  return std::make_unique<GroupNormOp>(*this);
}

std::vector<std::unique_ptr<Op>> GroupNormOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<GroupNormGradOp>(*this));
  return upops;
}

bool GroupNormOp::canBeReplacedByIdentity() const {
  return inInfo(getXInIndex()).nelms() == 0;
}

void GroupNormOp::setup() {
  // The input and output are of shape (N x C x H x W). If 4D input
  outInfo(getYOutIndex()) = inInfo(getXInIndex());

  // For each sample (dimension 0), and each group, there is a single mean and a
  // single inverse standard deviation
  outInfo(getInvStdDevOutIndex()) = {
      inInfo(getXInIndex()).dataType(),
      {inInfo(getXInIndex()).dim(0) * num_groups}};
  outInfo(getMeanOutIndex()) = {inInfo(getXInIndex()).dataType(),
                                {inInfo(getXInIndex()).dim(0) * num_groups}};
}

void GroupNormOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("num_groups", num_groups);
  os.appendAttribute("epsilon", epsilon);
}

GroupNormGradOp::GroupNormGradOp(const GroupNormOp &op_)
    : Op(Onnx::GradOperators::GroupNormalizationGrad, op_.getSettings()),
      epsilon(op_.getEpsilon()),
      fwdInInfo(op_.inInfo(GroupNormOp::getXInIndex())),
      fwdScaleInInfo(op_.inInfo(GroupNormOp::getScaleInIndex())),
      fwdBInInfo(op_.inInfo(GroupNormOp::getBInIndex())) {}

const std::map<int, int> &GroupNormGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getXGradOutIndex(), GroupNormOp::getXInIndex()},
      {getScaleOutIndex(), GroupNormOp::getScaleInIndex()},
      {getBOutIndex(), GroupNormOp::getBInIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &GroupNormGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getYGradInIndex(), GroupNormOp::getYOutIndex(), GradOpInType::GradOut},
      {getXInIndex(), GroupNormOp::getXInIndex(), GradOpInType::In},
      {getScaleInIndex(), GroupNormOp::getScaleInIndex(), GradOpInType::In},
      {getMeanInIndex(), GroupNormOp::getMeanOutIndex(), GradOpInType::Out},
      {getInvStdDevInIndex(),
       GroupNormOp::getInvStdDevOutIndex(),
       GradOpInType::Out}};
  return inInfo;
}

void GroupNormGradOp::setup() {
  TensorInfo xOutInfo = fwdInInfo;
  xOutInfo.set(xOutInfo.dataType(), inTensor(getXInIndex())->info.shape());

  outInfo(getXGradOutIndex()) = xOutInfo;
  outInfo(getScaleOutIndex()) = fwdScaleInInfo;
  outInfo(getBOutIndex())     = fwdBInInfo;
}

std::unique_ptr<Op> GroupNormGradOp::clone() const {
  return std::make_unique<GroupNormGradOp>(*this);
}

void GroupNormGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("epsilon", epsilon);
}
namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition groupNormOpDef(
    {OpDefinition::Inputs({
         {"X", T},
         {"Scale", T},
         {"Bias", T},
     }),
     OpDefinition::Outputs({{"Y", T}, {"Mean", T}, {"Var", T}}),
     OpDefinition::Attributes({{"num_groups", {"*"}}, {"epsilon", {"*"}}})});

static OpCreator<GroupNormOp> groupNormOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::GroupNormalization_1, groupNormOpDef},
    }),
    [](const OpCreatorInfo &info) {
      int64_t num_groups =
          info.attributes.getAttribute<Attributes::Int>("num_groups");

      // default epsilon is 10**(-5)
      float epsilon =
          info.attributes.getAttribute<Attributes::Float>("epsilon", 1e-5f);

      return std::unique_ptr<Op>(
          new GroupNormOp(info.opid, num_groups, epsilon, info.settings));
    },
    true);

} // namespace

} // namespace popart
