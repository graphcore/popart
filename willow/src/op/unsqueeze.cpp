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
  throw error("No gradient operations for unsqueeze is available. Unsqueeze "
              "should have been automatically replaced by a reshape operation "
              "by the built-in OpToReshape pattern");
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
    [](const OpCreatorInfo &info) {
      std::vector<int64_t> axes =
          info.attributes.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new UnsqueezeOp(info.opid, axes, info.settings));
    },
    true);
} // namespace

} // namespace popart
