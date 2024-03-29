// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <cstddef>
#include <map>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/shardingplan.hpp>

#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"

namespace popart {

ShardingPlan::ShardingPlan(ShardingMethod method_, ShardOpSettings settings_)
    : method(method_), shardSettings(settings_), totalNumShards(0) {}

ShardingPlan::ShardingPlan(ShardingMethod method_,
                           const ShardIdMap &shardMap_,
                           Graph &graph_,
                           ShardOpSettings settings_)
    : method(method_), shardIdMap(shardMap_), shardSettings(settings_),
      totalNumShards(0) {
  for (auto &idAndShardIds : shardIdMap) {
    fillInfoMapFromIdMap(idAndShardIds.first, graph_);
  }
}

ShardingPlan::ShardingPlan(ShardingMethod method_,
                           const ShardInfoMap &shardMap_,
                           ShardOpSettings settings_)
    : method(method_), shardInfoMap(shardMap_), shardSettings(settings_),
      totalNumShards(0) {}

void ShardingPlan::fillInfoMapFromIdMap(TensorId id, Graph &graph_) {
  auto &shardIds  = shardIdMap.at(id);
  TensorInfo info = graph_.getTensors().get(id)->info;
  std::vector<TensorInfo> shardInfos;
  shardInfos.reserve(shardIds.size());
  for (auto shardId : shardIds) {
    shardInfos.push_back(graph_.getTensors().get(shardId)->info);
  }
  shardInfoMap.insert({id, {id, info, shardInfos}});
}

void ShardingPlan::insertIdMap(const ShardIdMap &shardMap_, Graph &graph_) {
  for (auto &idAndShardIds : shardMap_) {
    shardIdMap.insert(idAndShardIds);
    fillInfoMapFromIdMap(idAndShardIds.first, graph_);
  }
}

void ShardingPlan::insertInfoMap(const ShardInfoMap &shardMap_) {
  for (auto &idAndInfos : shardMap_) {
    shardInfoMap.insert(idAndInfos);
  }
}

bool ShardingPlan::canDynamicShard() const {
  bool can = true;
  for (auto &idAndInfos : shardInfoMap) {
    for (size_t i = 1; i < idAndInfos.second.infos.size(); ++i) {
      // All sharded tensor infos must be equal
      if (idAndInfos.second.infos.at(i) != idAndInfos.second.infos.at(0)) {
        can = false;
      }
    }
  }
  return can;
}

bool ShardingPlan::canLoop() const { return canDynamicShard(); }

} // namespace popart
