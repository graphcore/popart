// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/remote.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

RemoteStoreOp::RemoteStoreOp(const OperatorIdentifier &_opid,
                             const Op::Settings &settings_,
                             RemoteBufferId rbid_)
    : Op(_opid, settings_), remotebuffer_id(rbid_) {}

std::unique_ptr<Op> RemoteStoreOp::clone() const {
  return std::make_unique<RemoteStoreOp>(*this);
}

void RemoteStoreOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("bufferid", remotebuffer_id);
}

RemoteLoadOp::RemoteLoadOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_,
                           RemoteBufferId rbid_)
    : Op(_opid, settings_), remotebuffer_id(rbid_) {}

std::unique_ptr<Op> RemoteLoadOp::clone() const {
  return std::make_unique<RemoteLoadOp>(*this);
}

void RemoteLoadOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("bufferid", remotebuffer_id);
}

void RemoteLoadOp::setup() {
  outInfo(getCachedTensorOutIndex()) = inInfo(getCachedTensorInIndex());
}

view::Regions RemoteLoadOp::modifies(InIndex index) const {
  if (index == getCachedTensorInIndex()) {
    return {view::Region::getFull(inShape(index), view::AccessType::Write)};
  } else if (index == getRemoteBufferOffsetInIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    throw error("Invalid index passed to RemoteLoadOp::modifies");
  }
}

view::Regions RemoteLoadOp::aliases(InIndex in, OutIndex) const {
  if (in == getCachedTensorInIndex()) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else if (in == getRemoteBufferOffsetInIndex()) {
    return {view::Region::getEmpty(inRank(in))};
  } else {
    throw error("Invalid index passed to RemoteLoadOp::aliases");
  }
}

view::RegMap RemoteLoadOp::fwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex == getRemoteBufferOffsetInIndex() &&
      output->hasIndex(getCachedTensorOutIndex())) {
    auto emptyRegion =
        view::Region::getEmpty(outRank(getCachedTensorOutIndex()));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap RemoteLoadOp::bwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex == getRemoteBufferOffsetInIndex() &&
      output->hasIndex(getCachedTensorOutIndex())) {
    auto emptyRegion =
        view::Region::getEmpty(inRank(getRemoteBufferOffsetInIndex()));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    remoteLoadOpDef({OpDefinition::Inputs({{"X", T}, {"O", {DataType::INT32}}}),
                     OpDefinition::Outputs({{"Y", T}}),
                     OpDefinition::Attributes({})});

static OpDefinition remoteStoreOpDef(
    {OpDefinition::Inputs({{"X", T}, {"O", {DataType::INT32}}}),
     OpDefinition::Outputs({}),
     OpDefinition::Attributes({})});

static OpCreator<RemoteLoadOp> remoteLoadOpCreator(
    OpDefinitions({{Onnx::CustomOperators::RemoteLoad, remoteLoadOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t bufferid =
          info.attributes.getAttribute<Attributes::Int>("bufferid");
      return std::unique_ptr<RemoteLoadOp>(
          new RemoteLoadOp(info.opid, info.settings, bufferid));
    },
    true);

static OpCreator<RemoteStoreOp> remoteStoreOpCreator(
    OpDefinitions({{Onnx::CustomOperators::RemoteStore, remoteStoreOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t bufferid =
          info.attributes.getAttribute<Attributes::Int>("bufferid");
      return std::unique_ptr<RemoteStoreOp>(
          new RemoteStoreOp(info.opid, info.settings, bufferid));
    },
    true);

} // namespace popart
