// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithmsinplace.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicbase.hpp>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

// Base Ops
DynamicBaseOp::DynamicBaseOp(const OperatorIdentifier &_opid,
                             std::vector<int64_t> axes_,
                             std::vector<int64_t> sizes_,
                             bool noOverlap_,
                             const Op::Settings &settings_)
    : Op(_opid, settings_), axes(axes_), sizes(sizes_), noOverlap(noOverlap_) {}

void DynamicBaseOp::setup() {}

void DynamicBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axes", axes);
  os.appendAttribute("sizes", sizes);
}

DynamicSliceBaseOp::DynamicSliceBaseOp(const OperatorIdentifier &_opid,
                                       std::vector<int64_t> axes_,
                                       std::vector<int64_t> sizes_,
                                       bool noOverlap_,
                                       const Op::Settings &settings_)
    : DynamicBaseOp(_opid, axes_, sizes_, noOverlap_, settings_) {}

void DynamicSliceBaseOp::setup() { outInfo(getOutIndex()) = createOutInfo(); }

TensorInfo DynamicSliceBaseOp::createOutInfo() const {
  auto in_info      = inInfo(getInIndex());
  auto output_shape = in_info.shape();

  for (size_t i = 0; i < axes.size(); ++i) {
    output_shape[axes[i]] = sizes[i];
  }

  return {in_info.dataType(), output_shape};
}

DynamicBinaryBaseOp::DynamicBinaryBaseOp(const OperatorIdentifier &_opid,
                                         std::vector<int64_t> axes_,
                                         std::vector<int64_t> sizes_,
                                         bool noOverlap_,
                                         const Op::Settings &settings_,
                                         TensorInfo updateInInfo_)
    : DynamicBaseOp(_opid, axes_, sizes_, noOverlap_, settings_),
      updateInInfo(updateInInfo_) {}

void DynamicBinaryBaseOp::setup() {
  if (input->hasIndex(getUpdateInIndex())) {
    updateInInfo = inInfo(getUpdateInIndex());
  }
  outInfo(getOutIndex()) = updateInInfo;
}

DynamicBinaryBaseInplaceOp::DynamicBinaryBaseInplaceOp(
    const OperatorIdentifier &_opid,
    std::vector<int64_t> axes_,
    std::vector<int64_t> sizes_,
    bool noOverlap_,
    const Op::Settings &settings_,
    TensorInfo updateInInfo_)
    : DynamicBinaryBaseOp(_opid,
                          axes_,
                          sizes_,
                          noOverlap_,
                          settings_,
                          updateInInfo_) {}

DynamicTernaryBaseOp::DynamicTernaryBaseOp(const OperatorIdentifier &_opid,
                                           std::vector<int64_t> axes_,
                                           std::vector<int64_t> sizes_,
                                           bool noOverlap_,
                                           const Op::Settings &settings_,
                                           TensorInfo updateInInfo_)
    : DynamicBinaryBaseOp(_opid,
                          axes_,
                          sizes_,
                          noOverlap_,
                          settings_,
                          updateInInfo_) {}

view::RegMap DynamicBinaryBaseInplaceOp::fwdRegMap(InIndex inIndex,
                                                   OutIndex outIndex) const {
  if (inIndex == getIndexInIndex()) {
    auto emptyRegion = view::Region::getEmpty(outRank(outIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap DynamicBinaryBaseInplaceOp::bwdRegMap(InIndex inIndex,
                                                   OutIndex outIndex) const {
  if (inIndex == getIndexInIndex()) {
    auto emptyRegion = view::Region::getEmpty(inRank(inIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

view::Regions DynamicBinaryBaseInplaceOp::aliases(InIndex in, OutIndex) const {
  if (in == getUpdateInIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

void DynamicBinaryBaseOp::growAliaser(PoprithmsAliaser &m) const {
  m.insertUnaryModifier(*this, getUpdateInIndex());
}

void DynamicBinaryBaseOp::setProposal(
    poprithms::memory::inplace::Proposal &proposal,
    const PoprithmsAliaser &aliaser,
    OperatorIdentifier opId) const {
  setProposalGate0(proposal, aliaser, opId);
}

// Modifies is the same as aliases
view::Regions DynamicBinaryBaseInplaceOp::modifies(InIndex index) const {
  return aliases(index, 0);
}

DynamicTernaryBaseInplaceOp::DynamicTernaryBaseInplaceOp(
    const OperatorIdentifier &_opid,
    std::vector<int64_t> axes_,
    std::vector<int64_t> sizes_,
    bool noOverlap_,
    const Op::Settings &settings_,
    TensorInfo updateInInfo_)
    : DynamicTernaryBaseOp(_opid,
                           axes_,
                           sizes_,
                           noOverlap_,
                           settings_,
                           updateInInfo_) {}

view::RegMap DynamicTernaryBaseInplaceOp::fwdRegMap(InIndex inIndex,
                                                    OutIndex outIndex) const {
  if (inIndex == getIndexInIndex() || inIndex == getInIndex()) {
    auto emptyRegion = view::Region::getEmpty(outRank(outIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap DynamicTernaryBaseInplaceOp::bwdRegMap(InIndex inIndex,
                                                    OutIndex outIndex) const {
  if (inIndex == getIndexInIndex() || inIndex == getInIndex()) {
    auto emptyRegion = view::Region::getEmpty(inRank(inIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

view::Regions DynamicTernaryBaseInplaceOp::aliases(InIndex in, OutIndex) const {
  if (in == getUpdateInIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

// Modifies is the same as aliases
view::Regions DynamicTernaryBaseInplaceOp::modifies(InIndex index) const {
  return aliases(index, 0);
}

} // namespace popart
