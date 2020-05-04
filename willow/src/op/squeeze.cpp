// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/squeeze.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

SqueezeBaseOp::SqueezeBaseOp(const OperatorIdentifier &_opid,
                             const std::vector<int64_t> &axes_,
                             const Op::Settings &settings_)
    : Op(_opid, settings_), axes(axes_) {}

void SqueezeBaseOp::setup() {
  if (axes.empty()) {
    setAxesToDefault();
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(),
                            squeeze(inShape(getInIndex()), axes)};
}

view::RegMap SqueezeBaseOp::fwdRegMap(InIndex inIndex,
                                      OutIndex outIndex) const {
  if (inIndex != 0 || outIndex != 0) {
    throw internal_error("[SqueezeBaseOp::fwdRegMap] "
                         "Received input index {} but only 0 allowed, "
                         "This for Op {}, ",
                         inIndex,
                         str());
  }
  // being conservative and returning the full region,
  // even for non-full input region :
  auto outRegion   = view::Region::getFull(outInfo(getOutIndex()).shape());
  auto emptyRegion = view::Region::getEmpty(outRank(getOutIndex()));
  return [emptyRegion, outRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return view::Regions(1, emptyRegion);
    }
    return view::Regions(1, outRegion);
  };
}

view::RegMap SqueezeBaseOp::bwdRegMap(InIndex inIndex,
                                      OutIndex outIndex) const {
  if (inIndex != 0 || outIndex != 0) {
    throw internal_error("[SqueezeBaseOp::bwdRegMap] "
                         "Received input index {} but only 0 allowed, "
                         "This for Op {}, ",
                         inIndex,
                         str());
  }
  auto inRegion    = view::Region::getFull(inInfo(getInIndex()).shape());
  auto emptyRegion = view::Region::getEmpty(inRank(getInIndex()));
  return [emptyRegion, inRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return view::Regions(1, emptyRegion);
    }
    return view::Regions(1, inRegion);
  };
}

void SqueezeBaseOp::setAxesToDefault() {
  auto in_shape = inShape(getInIndex());
  for (int i = 0; i < in_shape.size(); i++) {
    if (in_shape[i] == 1) {
      axes.push_back(i);
    }
  }
}

void SqueezeBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axes", axes);
}

SqueezeOp::SqueezeOp(const OperatorIdentifier &_opid,
                     const std::vector<int64_t> &axes_,
                     const Op::Settings &settings_)
    : SqueezeBaseOp(_opid, axes_, settings_) {}

std::vector<std::unique_ptr<Op>> SqueezeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SqueezeGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> SqueezeOp::clone() const {
  return std::make_unique<SqueezeOp>(*this);
}

std::unique_ptr<Op>
SqueezeOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SqueezeInplace) {
    return std::make_unique<SqueezeInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::vector<std::tuple<OperatorIdentifier, float>>
SqueezeOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::SqueezeInplace, 10}};
}

void SqueezeGradOp::setup() { outInfo(getOutIndex()) = unsqueezedInfo; }

std::unique_ptr<Op> SqueezeGradOp::clone() const {
  return std::make_unique<SqueezeGradOp>(*this);
}

SqueezeGradOp::SqueezeGradOp(const SqueezeOp &op_)
    : Op(Onnx::GradOperators::SqueezeGrad, op_.getSettings()),
      unsqueezedInfo(op_.inInfo(SqueezeOp::getInIndex())) {}

const std::vector<GradInOutMapper> &SqueezeGradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of squeeze
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SqueezeOp::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &SqueezeGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SqueezeOp::getInIndex()}};
  return outInfo;
}

SqueezeInplaceOp::SqueezeInplaceOp(const SqueezeOp &op)
    : SqueezeBaseOp(Onnx::CustomOperators::SqueezeInplace,
                    op.getAxes(),
                    op.settings) {}

std::unique_ptr<Op> SqueezeInplaceOp::clone() const {
  return std::make_unique<SqueezeInplaceOp>(*this);
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

static OpDefinition squeezeOpDef({OpDefinition::Inputs({{"data", T}}),
                                  OpDefinition::Outputs({{"squeezed", T}}),
                                  OpDefinition::Attributes({{"axes", {"*"}}})});

static OpCreator<SqueezeOp> squeezeOpCreator(
    OpDefinitions({
        {Onnx::Operators::Squeeze_1, squeezeOpDef},
        {Onnx::Operators::Squeeze_11, squeezeOpDef},
    }),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(new SqueezeOp(_opid, axes, settings));
    },
    true);
} // namespace

} // namespace popart
