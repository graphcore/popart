// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/shapeddropout.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ShapedDropoutOp::ShapedDropoutOp(const OperatorIdentifier &_opid,
                                 float ratio_,
                                 const std::vector<int64_t> &shape_,
                                 const Op::Settings &settings_)
    : DropoutBaseOp(_opid, ratio_, settings_), shape(shape_) {}

std::unique_ptr<Op> ShapedDropoutOp::clone() const {
  return std::make_unique<ShapedDropoutOp>(*this);
}

void ShapedDropoutOp::setup() {
  const auto &inputInfo = inInfo(getInIndex());

  if (!npBroadcastable(inputInfo.shape(), getShape())) {
    throw error(
        "ShapedDropout: incompatible input tensor and dropout shape. "
        "{} is configured with shape {} which is not broadcastable to the "
        "input tensor with shape {}.",
        opid,
        getShape(),
        inputInfo.shape());
  }

  outInfo(getOutIndex()) = inputInfo;
}

std::vector<std::unique_ptr<Op>> ShapedDropoutOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<ShapedDropoutGradOp>(*this));
  return upops;
}

void ShapedDropoutOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("ratio", getRatio());
  os.appendAttribute("shape", getShape());
}

ShapedDropoutGradOp::ShapedDropoutGradOp(const ShapedDropoutOp &fwdOp)
    : ShapedDropoutOp(fwdOp.opid,
                      fwdOp.getRatio(),
                      fwdOp.getShape(),
                      fwdOp.getSettings()) {
  // TODO: Do something.
}

std::unique_ptr<Op> ShapedDropoutGradOp::clone() const {
  return std::make_unique<ShapedDropoutGradOp>(*this);
}

const std::vector<GradInOutMapper> &ShapedDropoutGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), ShapedDropoutOp::getOutIndex(), GradOpInType::GradOut},
      // Dropout and DropoutGrad inherit from the same base op, so share the
      // same seed InIndex
      {getSeedInIndex(), ShapedDropoutOp::getSeedInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &ShapedDropoutGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ShapedDropoutOp::getInIndex()}};
  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    shapedDropoutOpDef({OpDefinition::Inputs({{"data", T}}),
                        OpDefinition::Outputs({{"output", T}}),
                        OpDefinition::Attributes({{"ratio", {"*"}}})});

static OpCreator<ShapedDropoutOp> shapedDropoutOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ShapedDropout_1,
                    shapedDropoutOpDef}}),
    [](const OpCreatorInfo &info) {
      float ratio = DropoutBaseOp::validateRatioAttribute(info);
      auto shape  = info.attributes.getAttribute<Attributes::Ints>("shape");
      return std::unique_ptr<Op>(
          new ShapedDropoutOp(info.opid, ratio, shape, info.settings));
    },
    true);

} // namespace
} // namespace popart
