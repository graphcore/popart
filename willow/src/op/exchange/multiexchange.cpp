// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

MultiExchangeOp::MultiExchangeOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_,
    const std::vector<ExchangeDescriptor> descriptors_)
    : ExchangeBaseOp(_opid, settings_), descriptors(descriptors_) {}

std::unique_ptr<Op> MultiExchangeOp::clone() const {
  return std::make_unique<MultiExchangeOp>(*this);
}

void MultiExchangeOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute(
      "exchangeDescriptors",
      logging::join(descriptors.begin(), descriptors.end(), ","));
}

void MultiExchangeOp::setup() {
  for (auto &out : output->tensorMap()) {
    for (auto &in : input->tensorMap()) {
      auto outIndices = outIndexToDescriptorIndex(out.first);
      auto inIndices  = inIndexToDescriptorIndex(in.first);
      if (outIndices.first == inIndices.first && inIndices.second == 0) {
        outInfo(out.first) = inInfo(in.first);
        logging::op::trace("[MultiExchangeOp] Propagating descriptor {}:{} "
                           "TensorInfo {}:{} ({}) -> {}:{} ({})",
                           inIndices.first,
                           getExchangeDescriptor(inIndices.first),
                           in.first,
                           in.second->id,
                           inInfo(in.first),
                           out.first,
                           out.second->id,
                           outInfo(out.first));
      }
    }
  }
}

view::Regions MultiExchangeOp::modifies(InIndex in) const {
  auto indices = inIndexToDescriptorIndex(in);
  auto isLoad  = getExchangeDescriptor(indices.first).getDirection() ==
                ExchangeDirection::Load;

  if (indices.second == 0 && isLoad) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::Regions MultiExchangeOp::aliases(InIndex in, OutIndex out) const {
  auto inIndices  = inIndexToDescriptorIndex(in);
  auto outIndices = outIndexToDescriptorIndex(out);

  if (inIndices.first == outIndices.first && inIndices.second == 0) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::RegMap MultiExchangeOp::fwdRegMap(InIndex inIndex,
                                        OutIndex outIndex) const {
  auto inIndices  = inIndexToDescriptorIndex(inIndex);
  auto outIndices = outIndexToDescriptorIndex(outIndex);

  if (inIndices.first != outIndices.first || inIndices.second != 0) {
    auto emptyRegion = view::Region::getEmpty(outRank(outIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap MultiExchangeOp::bwdRegMap(InIndex inIndex,
                                        OutIndex outIndex) const {
  auto inIndices  = inIndexToDescriptorIndex(inIndex);
  auto outIndices = outIndexToDescriptorIndex(outIndex);

  if (inIndices.first != outIndices.first || inIndices.second != 0) {
    auto emptyRegion = view::Region::getEmpty(inRank(inIndex));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

int MultiExchangeOp::numLoads() const {
  return std::count_if(
      descriptors.begin(), descriptors.end(), [](const ExchangeDescriptor &ed) {
        return ed.getDirection() == ExchangeDirection::Load;
      });
}

int MultiExchangeOp::numStores() const {
  return std::count_if(
      descriptors.begin(), descriptors.end(), [](const ExchangeDescriptor &ed) {
        return ed.getDirection() == ExchangeDirection::Store;
      });
}

VGraphIdAndTileSet MultiExchangeOp::getIntrospectionInVirtualGraphId(
    InIndex in,
    std::set<OpId> visited) const {
  auto descIndex = inIndexToDescriptorIndex(in);
  auto vgid      = descriptors.at(descIndex.first).getVGraphID();
  auto tiles     = descriptors.at(descIndex.first).getTileSet();
  return {vgid ? *vgid : unusedVGraphId, tiles};
}

VGraphIdAndTileSet MultiExchangeOp::getIntrospectionOutVirtualGraphId(
    OutIndex out,
    std::set<OpId> visited) const {
  auto descIndex = outIndexToDescriptorIndex(out);
  auto vgid      = descriptors.at(descIndex.first).getVGraphID();
  auto tiles     = descriptors.at(descIndex.first).getTileSet();
  return {vgid ? *vgid : unusedVGraphId, tiles};
}

ReplicatedTensorShardingIndices
MultiExchangeOp::getReplicatedTensorShardingIndices() const {
  ReplicatedTensorShardingIndices indices;

  InIndex inIdx   = 0;
  OutIndex outIdx = 0;

  for (auto &d : descriptors) {
    indices.insert(
        d.getNumOutputs() > 1
            ? (std::pair<std::set<InIndex>, std::set<OutIndex>>{{inIdx},
                                                                {outIdx}})
            : std::pair<std::set<InIndex>, std::set<OutIndex>>({{inIdx}, {}}));
    inIdx += d.getNumInputs();
    outIdx += d.getNumOutputs();
  }

  return indices;
}

ExchangeDescriptor MultiExchangeOp::getExchangeDescriptor(int index) const {
  return descriptors.at(index);
}

std::pair<int, int>
MultiExchangeOp::inIndexToDescriptorIndex(InIndex index) const {
  int descIndex = 0;
  int count     = 0;
  for (auto &d : descriptors) {
    if (index >= count && index < count + d.getNumInputs()) {
      return {descIndex, index - count};
    }
    count += d.getNumInputs();
    ++descIndex;
  }
  throw error(
      "[MultiExchangeOp] No descriptor for input index {} ({} descriptors)",
      index,
      descriptors.size());
}

std::pair<int, int>
MultiExchangeOp::outIndexToDescriptorIndex(OutIndex index) const {
  int descIndex = 0;
  int count     = 0;
  for (auto &d : descriptors) {
    if (index >= count && index < count + d.getNumOutputs()) {
      return {descIndex, index - count};
    }
    count += d.getNumOutputs();
    ++descIndex;
  }
  throw error(
      "[MultiExchangeOp] No descriptor for output index {} ({} descriptors)",
      index,
      descriptors.size());
}

std::vector<InIndex>
MultiExchangeOp::descriptorIndexToInIndices(int index) const {
  int offset = std::accumulate(descriptors.begin(),
                               descriptors.begin() + index,
                               0,
                               [](const int v, const ExchangeDescriptor &d) {
                                 return v + d.getNumInputs();
                               });
  std::vector<InIndex> indices(descriptors.at(index).getNumInputs());
  std::iota(indices.begin(), indices.end(), offset);
  return indices;
}

std::vector<OutIndex>
MultiExchangeOp::descriptorIndexToOutIndices(int index) const {
  int offset = std::accumulate(descriptors.begin(),
                               descriptors.begin() + index,
                               0,
                               [](const int v, const ExchangeDescriptor &d) {
                                 return v + d.getNumOutputs();
                               });
  std::vector<InIndex> indices(descriptors.at(index).getNumOutputs());
  std::iota(indices.begin(), indices.end(), offset);
  return indices;
}

} // namespace popart
