// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/reverse.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

void ReverseBaseOp::setup() {
  // Validate dimensions
  auto rank = inRank(getInIndex());
  std::vector<bool> reverses(rank, false);
  for (auto dim : dimensions) {
    if (dim < 0) {
      throw error("ReverseOp '{}' invalid dimension '{}'. Only positive "
                  "dimensions are supported",
                  str(),
                  dim);
    }

    if (dim >= rank) {
      throw error(
          "ReverseOp '{}' invalid dimension '{}' for input tensor of rank {}",
          str(),
          dim,
          rank);
    }

    if (reverses[dim] == true) {
      throw error("Dimension {} appears multiple times in the list of "
                  "dimensions of ReverseOp '{}'",
                  dim,
                  str());
    }

    reverses[dim] = true;
  }

  outInfo(getOutIndex()) = inInfo(getInIndex());
}

bool ReverseBaseOp::canBeReplacedByIdentity() const {
  return dimensions.size() == 0;
}

std::vector<std::unique_ptr<Op>> ReverseOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<ReverseGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> ReverseOp::clone() const {
  return std::make_unique<ReverseOp>(*this);
}

void ReverseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("dimensions", getDimensions());
}

view::RegMap ReverseBaseOp::fwdRegMap(InIndex inIndex, OutIndex) const {
  if (inIndex != 0) {
    throw internal_error("[ReverseBaseOp::*wdRegMap] "
                         "Received input index {} but only 0 allowed, "
                         "This for Op {}, ",
                         inIndex,
                         str());
  }
  auto emptyRegion = view::Region::getEmpty(outRank(getOutIndex()));
  auto shape       = inShape(getInIndex());
  auto dimensions  = getDimensions();
  return [dimensions, shape, emptyRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return view::Regions(1, emptyRegion);
    }
    return view::Regions(1, r.reverse(shape, dimensions));
  };
}

view::RegMap ReverseBaseOp::bwdRegMap(InIndex inIndex,
                                      OutIndex outIndex) const {
  // same as fwd
  return fwdRegMap(inIndex, outIndex);
}

ReverseGradOp::ReverseGradOp(const ReverseOp &op_)
    : ReverseOp(Onnx::GradOperators::ReverseGrad,
                op_.getSettings(),
                op_.getDimensions()) {}

const std::vector<GradInOutMapper> &ReverseGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReverseOp::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &ReverseGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ReverseOp::getInIndex()}};
  return outInfo;
}

std::unique_ptr<Op> ReverseInplaceOp::clone() const {
  return std::make_unique<ReverseInplaceOp>(*this);
}

std::unique_ptr<Op>
ReverseOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ReverseInplace) {
    return std::make_unique<ReverseInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
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
    reverseOpDef({OpDefinition::Inputs({{"data", T}}),
                  OpDefinition::Outputs({{"reversed", T}}),
                  OpDefinition::Attributes({{"dimensions", {"*"}}})});

static OpCreator<ReverseOp> reverseOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Reverse, reverseOpDef}}),
    [](const OpCreatorInfo &info) {
      std::vector<int64_t> dimensions =
          info.attributes.getAttribute<Attributes::Ints>("dimensions", {});

      return std::unique_ptr<Op>(
          new ReverseOp(info.opid, info.settings, dimensions));
    },
    true);

} // namespace

} // namespace popart