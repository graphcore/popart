
#include <memory>
#include <poponnx/op/groupnorm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

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

void GroupNormOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
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
      {getYGradInIndex(), GroupNormOp::getYOutIndex(), GradOpInType::GRADOUT},
      {getXInIndex(), GroupNormOp::getXInIndex(), GradOpInType::IN},
      {getScaleInIndex(), GroupNormOp::getScaleInIndex(), GradOpInType::IN},
      {getMeanInIndex(), GroupNormOp::getMeanOutIndex(), GradOpInType::OUT},
      {getInvStdDevInIndex(),
       GroupNormOp::getInvStdDevOutIndex(),
       GradOpInType::OUT}};
  return inInfo;
}

void GroupNormGradOp::setup() {

  outInfo(getXGradOutIndex()) = fwdInInfo;
  outInfo(getScaleOutIndex()) = fwdScaleInInfo;
  outInfo(getBOutIndex())     = fwdBInInfo;
}

std::unique_ptr<Op> GroupNormGradOp::clone() const {
  return std::make_unique<GroupNormGradOp>(*this);
}

void GroupNormGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("epsilon", epsilon);
}
namespace {
static OpCreator<GroupNormOp> groupNormOpCreator(
    Onnx::CustomOperators::GroupNormalization_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t num_groups = attr.getAttribute<Attributes::Int>("num_groups");

      // default epsilon is 10**(-5)
      float epsilon = attr.getAttribute<Attributes::Float>("epsilon", 1e-5f);

      return std::unique_ptr<Op>(
          new GroupNormOp(_opid, num_groups, epsilon, settings));
    },
    true);

} // namespace

} // namespace poponnx
