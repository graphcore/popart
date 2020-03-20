// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/unsqueeze.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

UnsqueezeOp::UnsqueezeOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &axes_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), axes(axes_) {}

std::vector<std::unique_ptr<Op>> UnsqueezeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<UnsqueezeGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> UnsqueezeOp::clone() const {
  return std::make_unique<UnsqueezeOp>(*this);
}

void UnsqueezeOp::setup() {
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(),
                            unsqueeze(inShape(getInIndex()), axes)};
}

void UnsqueezeOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axes", axes);
}

void UnsqueezeGradOp::setup() { outInfo(getOutIndex()) = squeezedInfo; }

std::unique_ptr<Op> UnsqueezeGradOp::clone() const {
  return std::make_unique<UnsqueezeGradOp>(*this);
}

UnsqueezeGradOp::UnsqueezeGradOp(const UnsqueezeOp &op_)
    : Op(Onnx::GradOperators::UnsqueezeGrad, op_.getSettings()),
      squeezedInfo(op_.inInfo(UnsqueezeOp::getInIndex())) {}

const std::vector<GradInOutMapper> &UnsqueezeGradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of unsqueeze
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), UnsqueezeOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &UnsqueezeGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), UnsqueezeOp::getInIndex()}};
  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};

static OpDefinition
    unsqueezeOpDef({OpDefinition::Inputs({{"data", T}}),
                    OpDefinition::Outputs({{"expanded", T}}),
                    OpDefinition::Attributes({{"axes", {"*"}}})});

static OpCreator<UnsqueezeOp> unsqueezeOpCreator(
    OpDefinitions({
        {Onnx::Operators::Unsqueeze_1, unsqueezeOpDef},
        {Onnx::Operators::Unsqueeze_11, unsqueezeOpDef},
    }),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(new UnsqueezeOp(_opid, axes, settings));
    },
    true);
} // namespace

} // namespace popart
