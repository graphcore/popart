// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <gcl/CollectiveBalancedReorder.hpp>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <popart/popx/replicatedtensorshardingbundle.hpp>

#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensornames.hpp"

namespace popart {
class Ir;

namespace popx {

ReplicatedTensorShardingBundle::ReplicatedTensorShardingBundle(const Ir &ir)
    : replicatedTensorShardingTracer(ir), cbrCounter(0) {}

void ReplicatedTensorShardingBundle::setCollectiveBalancedReorder(
    const TensorId &tensorId,
    CollectiveBalancedReorderId cbrId) {
  collectiveReorderIds[tensorId] = cbrId;
}

CollectiveBalancedReorderId
ReplicatedTensorShardingBundle::registerCollectiveBalancedReorder(
    std::shared_ptr<gcl::CollectiveBalancedReorder> cbr) {
  auto cbrId                = getAndIncrCBRCounter();
  collectiveReorders[cbrId] = cbr;
  return cbrId;
}

bool ReplicatedTensorShardingBundle::hasCollectiveBalancedReorder(
    const TensorId &tensorId) const {
  auto remoteArgId = getRemoteArgTensorId(stripAllReservedPrefixes(tensorId));
  {
    auto it = collectiveReorderIds.find(tensorId);
    if (it != collectiveReorderIds.end()) {
      return true;
    }
  }
  {
    auto it = collectiveReorderIds.find(remoteArgId);
    if (it != collectiveReorderIds.end()) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<gcl::CollectiveBalancedReorder>
ReplicatedTensorShardingBundle::getCollectiveBalancedReorder(
    const TensorId &tensorId) const {
  auto remoteArgId = getRemoteArgTensorId(stripAllReservedPrefixes(tensorId));
  {
    auto it = collectiveReorderIds.find(tensorId);
    if (it != collectiveReorderIds.end()) {
      return collectiveReorders.at(it->second);
    }
  }
  {
    auto it = collectiveReorderIds.find(remoteArgId);
    if (it != collectiveReorderIds.end()) {
      return collectiveReorders.at(it->second);
    }
  }
  throw internal_error(
      "[ReplicatedTensorShardingState::getCollectiveBalancedReorder] Could not "
      "find CBR for {} or {}",
      tensorId,
      remoteArgId);
}

CollectiveBalancedReorderId
ReplicatedTensorShardingBundle::getAndIncrCBRCounter() {
  CollectiveBalancedReorderId cbrId = cbrCounter;
  ++cbrCounter;
  return cbrId;
}

const gcl::CollectiveBalancedHostRearrangement &
ReplicatedTensorShardingBundle::getCollectiveBalancedHostRearrangement(
    const TensorId &tensorId) const {

  return getCollectiveBalancedReorder(tensorId)->getHostRearrangement();
}

} // namespace popx
} // namespace popart
