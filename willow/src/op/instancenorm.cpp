#include <algorithm>
#include <vector>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/instancenorm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

InstanceNormOp::InstanceNormOp(const OperatorIdentifier &_opid,
                               float _epsilon,
                               const Op::Settings &settings_)
    : Op(_opid, settings_), epsilon(_epsilon) {}

std::unique_ptr<Op> InstanceNormOp::clone() const {
  return make_unique<InstanceNormOp>(*this);
}

std::vector<std::unique_ptr<Op>> InstanceNormOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<InstanceNormGradOp>(*this));
  return upops;
}

void InstanceNormOp::setup() {
  auto input_info  = inInfo(getInputInIndex());
  auto input_shape = input_info.shape();
  auto batch_size  = input_shape[0];
  auto features    = input_shape[1];

  outInfo(getOutIndex()) = input_info;

  createAndConnectOutTensor(getMeanOutIndex(),
                            outTensor(getOutIndex())->id + "_mean");
  outInfo(getMeanOutIndex()) = {input_info.dataType(), {batch_size * features}};

  createAndConnectOutTensor(getInvStdDevOutIndex(),
                            outTensor(getOutIndex())->id + "_invStdDev");
  outInfo(getInvStdDevOutIndex()) = {input_info.dataType(),
                                     Shape{batch_size * features}};
}

void InstanceNormOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("epsilon", epsilon);
}

InstanceNormGradOp::InstanceNormGradOp(const InstanceNormOp &fwd_op)
    : Op(Onnx::GradOperators::InstanceNormalizationGrad, fwd_op.getSettings()) {
}

const std::vector<GradInOutMapper> &InstanceNormGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInputInIndex(), InstanceNormOp::getInputInIndex(), GradOpInType::IN},
      {getScaleInIndex(), InstanceNormOp::getScaleInIndex(), GradOpInType::IN},
      {getOutGradInIndex(),
       InstanceNormOp::getInputInIndex(),
       GradOpInType::GRADOUT},
      {getMeanInIndex(), InstanceNormOp::getMeanOutIndex(), GradOpInType::OUT},
      {getInvStdDevInIndex(),
       InstanceNormOp::getInvStdDevOutIndex(),
       GradOpInType::OUT},
  };

  return inInfo;
}

const std::map<int, int> &InstanceNormGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getInputOutIndex(), InstanceNormOp::getInputInIndex()},
      {getScaleOutIndex(), InstanceNormOp::getScaleInIndex()},
      {getBOutIndex(), InstanceNormOp::getBInIndex()}};
  return outInfo;
}

void InstanceNormGradOp::setup() {
  const auto in_info  = inInfo(getOutGradInIndex());
  const auto in_type  = in_info.dataType();
  const auto in_shape = in_info.shape();

  outInfo(getInputOutIndex()) = in_info;
  outInfo(getScaleOutIndex()) = {in_type, {in_shape[1]}};
  outInfo(getBOutIndex())     = {in_type, {in_shape[1]}};
}

namespace {
static OpCreator<InstanceNormOp> instanceNormOpCreator(
    Onnx::Operators::InstanceNormalization_6,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      // default epsilon is 10**(-5)
      float epsilon = attr.getAttribute<Attributes::Float>("epsilon", 1e-5f);

      return std::unique_ptr<Op>(new InstanceNormOp(_opid, epsilon, settings));
    },
    true);

} // namespace

} // namespace poponnx
