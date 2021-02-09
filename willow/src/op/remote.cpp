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

ReplicatedTensorShardingIndices
RemoteStoreOp::getReplicatedTensorShardingIndices() const {
  return {{{RemoteStoreOp::getLocalTensorInIndex()}, {}}};
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
  outInfo(getLocalTensorOutIndex()) = inInfo(getLocalTensorInIndex());
}

view::Regions RemoteLoadOp::modifies(InIndex index) const {
  if (index == getLocalTensorInIndex()) {
    return {view::Region::getFull(inShape(index), view::AccessType::Write)};
  } else if (index == getRemoteBufferOffsetInIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    throw error("Invalid index passed to RemoteLoadOp::modifies");
  }
}

view::Regions RemoteLoadOp::aliases(InIndex in, OutIndex) const {
  if (in == getLocalTensorInIndex()) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else if (in == getRemoteBufferOffsetInIndex()) {
    return {view::Region::getEmpty(inRank(in))};
  } else {
    throw error("Invalid index passed to RemoteLoadOp::aliases");
  }
}

view::RegMap RemoteLoadOp::fwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex == getRemoteBufferOffsetInIndex() &&
      output->hasIndex(getLocalTensorOutIndex())) {
    auto emptyRegion =
        view::Region::getEmpty(outRank(getLocalTensorOutIndex()));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap RemoteLoadOp::bwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex == getRemoteBufferOffsetInIndex() &&
      output->hasIndex(getLocalTensorOutIndex())) {
    auto emptyRegion =
        view::Region::getEmpty(inRank(getRemoteBufferOffsetInIndex()));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

ReplicatedTensorShardingIndices
RemoteLoadOp::getReplicatedTensorShardingIndices() const {
  return {{{RemoteLoadOp::getLocalTensorInIndex()},
           {RemoteLoadOp::getLocalTensorOutIndex()}}};
}

RemoteExchangeOp::RemoteExchangeOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_,
    const std::vector<RemoteBufferId> remotebuffer_ids_,
    const std::vector<std::pair<OptionalVGraphId, TileSet>> vgidAndTiles_)
    : Op(_opid, settings_), remotebufferIds(remotebuffer_ids_),
      vgidAndTiles(vgidAndTiles_) {}

std::unique_ptr<Op> RemoteExchangeOp::clone() const {
  return std::make_unique<RemoteExchangeOp>(*this);
}

void RemoteExchangeOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("bufferids", remotebufferIds);

  std::vector<VGraphId> vgids;
  vgids.reserve(vgidAndTiles.size());
  std::vector<int64_t> tileSets;
  tileSets.reserve(vgidAndTiles.size());

  for (auto &vgid : vgidAndTiles) {
    vgids.push_back(vgid.first ? *vgid.first : unusedVGraphId);
    tileSets.push_back(static_cast<int64_t>(vgid.second));
  }

  os.appendAttribute("vgids", vgids);
  os.appendAttribute("tileSets", tileSets);
}

void RemoteExchangeOp::setup() {
  for (auto &out : output->tensorMap()) {
    outInfo(out.first) = inInfo(out.first);
  }
}

view::Regions RemoteExchangeOp::modifies(InIndex in) const {
  if (in < numLoads()) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::Regions RemoteExchangeOp::aliases(InIndex in, OutIndex out) const {
  if (in == out) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::RegMap RemoteExchangeOp::fwdRegMap(InIndex inIndex,
                                         OutIndex outIndex) const {
  if (inIndex != outIndex) {
    auto emptyRegion = view::Region::getEmpty(outRank(outIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap RemoteExchangeOp::bwdRegMap(InIndex inIndex,
                                         OutIndex outIndex) const {
  if (inIndex != outIndex) {
    auto emptyRegion = view::Region::getEmpty(inRank(inIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

int RemoteExchangeOp::numLoads() const {
  // Each load produces one Op output
  return output->tensorMap().size();
}

int RemoteExchangeOp::numStores() const {
  // Each store and load has 2 inputs
  return (input->tensorMap().size() / 2) - numLoads();
}

VGraphIdAndTileSet RemoteExchangeOp::getIntrospectionInVirtualGraphId(
    InIndex in,
    std::set<OpId> visited) const {
  auto vgid = vgidAndTiles.at(in % (numLoads() + numStores()));
  return {vgid.first ? *vgid.first : unusedVGraphId, vgid.second};
}

VGraphIdAndTileSet RemoteExchangeOp::getIntrospectionOutVirtualGraphId(
    OutIndex out,
    std::set<OpId> visited) const {
  auto vgid = vgidAndTiles.at(out % (numLoads() + numStores()));
  return {vgid.first ? *vgid.first : unusedVGraphId, vgid.second};
}

ReplicatedTensorShardingIndices
RemoteExchangeOp::getReplicatedTensorShardingIndices() const {
  ReplicatedTensorShardingIndices indices;

  for (auto &out : output->tensorMap()) {
    indices.insert({{out.first}, {out.first}});
  }

  for (InIndex inIndex = numLoads(); inIndex < numLoads() + numStores();
       ++inIndex) {
    indices.insert({{inIndex}, {}});
  }

  return indices;
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
