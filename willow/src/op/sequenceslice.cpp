// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/sequenceslice.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

#include <popart/alias/aliasmodel.hpp>

namespace popart {

SequenceSliceOp::SequenceSliceOp(const OperatorIdentifier &opid_,
                                 bool zeroUnused_,
                                 const Op::Settings &settings_)
    : Op(opid_, settings_), zeroUnused(zeroUnused_) {}

std::unique_ptr<Op> SequenceSliceOp::clone() const {
  return std::make_unique<SequenceSliceOp>(*this);
}

std::vector<std::unique_ptr<Op>> SequenceSliceOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  return upops;
}

void SequenceSliceOp::setup() {
  outInfo(getOutIndex()) = inInfo(getDestinationInIndex());
}

std::unique_ptr<Op> SequenceSliceOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SequenceSliceInplace) {
    return std::make_unique<SequenceSliceInplaceOp>(
        Onnx::CustomOperators::SequenceSliceInplace, zeroUnused, settings);
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

void SequenceSliceOp::growAliasModel(AliasModel &m) const {
  m.insertUnaryModifier(*this, getDestinationInIndex());
}

poprithms::memory::inplace::Proposal
SequenceSliceOp::mapInplaceProposal(const AliasModel &aliasModel,
                                    OperatorIdentifier opId) const {
  return mapInplaceProposalGate0(aliasModel, opId);
}

std::vector<std::tuple<OperatorIdentifier, float>>
SequenceSliceOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::SequenceSliceInplace, 10.f}};
}

SequenceSliceInplaceOp::SequenceSliceInplaceOp(const OperatorIdentifier &opid_,
                                               bool zeroUnused,
                                               const Op::Settings &settings_)
    : SequenceSliceOp(opid_, zeroUnused, settings_) {}

std::unique_ptr<Op> SequenceSliceInplaceOp::clone() const {
  return std::make_unique<SequenceSliceInplaceOp>(*this);
}

view::RegMap SequenceSliceInplaceOp::fwdRegMap(InIndex inIndex,
                                               OutIndex outIndex) const {
  if (inIndex == getDestinationInIndex()) {
    return [](const view::Region &r) { return view::Regions(1, r); };
  } else {
    auto emptyRegion = view::Region::getEmpty(outRank(outIndex));
    return [emptyRegion](const view::Region &r) {
      return view::Regions(1, emptyRegion);
    };
  }
}

view::RegMap SequenceSliceInplaceOp::bwdRegMap(InIndex inIndex,
                                               OutIndex outIndex) const {
  if (inIndex == getDestinationInIndex()) {
    return [](const view::Region &r) { return view::Regions(1, r); };
  } else {
    auto emptyRegion = view::Region::getEmpty(outRank(outIndex));
    return [emptyRegion](const view::Region &r) {
      return view::Regions(1, emptyRegion);
    };
  }
}

view::Regions SequenceSliceInplaceOp::aliases(InIndex inIndex, OutIndex) const {
  if (inIndex == getDestinationInIndex()) {
    return {view::Region::getFull(inShape(inIndex))};
  } else {
    return {view::Region::getEmpty(inRank(inIndex))};
  }
}

// Modifies is the same as aliases
view::Regions SequenceSliceInplaceOp::modifies(InIndex index) const {
  return aliases(index, 0);
}

// Creators
static OpDefinition::DataTypes U = {DataType::UINT32};

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    sequenceSliceOpDef({OpDefinition::Inputs({{"Source", T},
                                              {"Destination", T},
                                              {"N", U},
                                              {"SourceOffset", U},
                                              {"DestinationOffset", U}}),
                        OpDefinition::Outputs({{"Result", T}}),
                        OpDefinition::Attributes({{"zeroUnused", {"*"}}})});

static OpCreator<SequenceSliceOp> sequenceSliceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::SequenceSlice_1,
                    sequenceSliceOpDef}}),
    [](const OpCreatorInfo &info) {
      bool zeroUnused =
          info.attributes.hasAttribute("zeroUnused") &&
          info.attributes.getAttribute<Attributes::Int>("zeroUnused");
      return std::unique_ptr<SequenceSliceOp>(
          new SequenceSliceOp(info.opid, zeroUnused, info.settings));
    },
    true);

} // namespace popart
