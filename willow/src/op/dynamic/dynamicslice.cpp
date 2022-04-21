// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

DynamicSliceOp::DynamicSliceOp(const OperatorIdentifier &_opid,
                               std::vector<int64_t> axes_,
                               std::vector<int64_t> sizes_,
                               bool noOverlap_,
                               const Op::Settings &settings_)
    : DynamicSliceBaseOp(_opid, axes_, sizes_, noOverlap_, settings_) {}

std::unique_ptr<Op> DynamicSliceOp::clone() const {
  return std::make_unique<DynamicSliceOp>(*this);
}

std::vector<std::unique_ptr<Op>> DynamicSliceOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<DynamicSlicePadGradOp>(
      Onnx::CustomGradOperators::DynamicSlicePadGrad,
      getAxes(),
      getSizes(),
      noOverlap,
      settings,
      inInfo(DynamicSliceBaseOp::getInIndex())));
  return upops;
}

std::vector<std::tuple<OperatorIdentifier, float>>
DynamicSliceOp::inplacePriorityDefault() const {
  if (hasInput(DynamicSliceOp::getSliceInIndex())) {
    // Inplace is only supported if the slice input to overwrite is available
    return {{Onnx::CustomOperators::DynamicSliceInplace_1, 10}};
  } else {
    return {};
  }
}

std::unique_ptr<Op>
DynamicSliceOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::DynamicSliceInplace_1) {
    return std::make_unique<DynamicSliceInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

void DynamicSliceOp::growAliasModel(AliasModel &m) const {
  if (hasInput(getSliceInIndex())) {
    m.insertUnaryModifier(*this, getSliceInIndex());
  } else {
    Op::growAliasModel(m);
  }
}

poprithms::memory::inplace::Proposal
DynamicSliceOp::mapInplaceProposal(const AliasModel &aliasModel,
                                   OperatorIdentifier id) const {
  if (hasInput(DynamicSliceOp::getSliceInIndex())) {
    return mapInplaceProposalGate0(aliasModel, id);
  }
  return Op::mapInplaceProposal(aliasModel, id);
}

DynamicSliceInplaceOp::DynamicSliceInplaceOp(const OperatorIdentifier &_opid,
                                             std::vector<int64_t> axes_,
                                             std::vector<int64_t> sizes_,
                                             bool noOverlap_,
                                             const Op::Settings &settings_)
    : DynamicSliceOp(_opid, axes_, sizes_, noOverlap_, settings_) {}

DynamicSliceInplaceOp::DynamicSliceInplaceOp(
    const DynamicSliceOp &dynamicSliceOp)
    : DynamicSliceOp(Onnx::CustomOperators::DynamicSliceInplace_1,
                     dynamicSliceOp.getAxes(),
                     dynamicSliceOp.getSizes(),
                     dynamicSliceOp.isNotOverlapping(),
                     dynamicSliceOp.getSettings()) {}

std::unique_ptr<Op> DynamicSliceInplaceOp::clone() const {
  return std::make_unique<DynamicSliceInplaceOp>(*this);
}

std::vector<std::tuple<OperatorIdentifier, float>>
DynamicSliceInplaceOp::inplacePriorityDefault() const {
  return {};
}

std::unique_ptr<Op>
DynamicSliceInplaceOp::getInplaceVariant(const OperatorIdentifier &o) const {
  // this throws an error
  return Op::getInplaceVariant(o);
}

view::RegMap DynamicSliceInplaceOp::fwdRegMap(InIndex inIndex,
                                              OutIndex outIndex) const {
  if (inIndex != getSliceInIndex()) {
    auto emptyRegion = view::Region::getEmpty(outRank(outIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap DynamicSliceInplaceOp::bwdRegMap(InIndex inIndex,
                                              OutIndex outIndex) const {
  if (inIndex != getSliceInIndex()) {
    auto emptyRegion = view::Region::getEmpty(inRank(inIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

view::Regions DynamicSliceInplaceOp::modifies(InIndex in) const {
  if (in == getSliceInIndex()) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::Regions DynamicSliceInplaceOp::aliases(InIndex in, OutIndex) const {
  if (in == getSliceInIndex()) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

// Grad Ops
DynamicSlicePadGradOp::DynamicSlicePadGradOp(const OperatorIdentifier &_opid,
                                             std::vector<int64_t> axes_,
                                             std::vector<int64_t> sizes_,
                                             bool noOverlap_,
                                             const Op::Settings &settings_,
                                             TensorInfo updateInInfo_)
    : DynamicBaseOp(_opid, axes_, sizes_, noOverlap_, settings_),
      updateInInfo(updateInInfo_) {}

void DynamicSlicePadGradOp::setup() { outInfo(getOutIndex()) = updateInInfo; }

std::unique_ptr<Op> DynamicSlicePadGradOp::clone() const {
  return std::make_unique<DynamicSlicePadGradOp>(*this);
}

const std::vector<GradInOutMapper> &
DynamicSlicePadGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), DynamicBaseOp::getOutIndex(), GradOpInType::GradOut},
      {getIndexInIndex(), DynamicBaseOp::getIndexInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &DynamicSlicePadGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DynamicSliceOp::getInIndex()}};
  return outInfo;
}

// Creators
static OpDefinition::DataTypes U = {DataType::UINT32};

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    dynamicSliceOpDef({OpDefinition::Inputs({{"X", T}, {"O", U}}),
                       OpDefinition::Outputs({{"Y", T}}),
                       OpDefinition::Attributes({
                           {"axes", {"*"}},
                           {"size", {"*"}},
                           {"noOverlap", {"*"}},
                       })});

static OpCreator<DynamicSliceBaseOp> dynamicSliceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::DynamicSlice_1, dynamicSliceOpDef}}),
    [](const OpCreatorInfo &info) {
      std::vector<int64_t> axes =
          info.attributes.getAttribute<Attributes::Ints>("axes");
      std::vector<int64_t> sizes =
          info.attributes.getAttribute<Attributes::Ints>("sizes");
      bool noOverlap =
          info.attributes.hasAttribute("noOverlap") &&
          info.attributes.getAttribute<Attributes::Int>("noOverlap");
      return std::unique_ptr<DynamicSliceBaseOp>(
          new DynamicSliceOp(info.opid, axes, sizes, noOverlap, info.settings));
    },
    true);

} // namespace popart
