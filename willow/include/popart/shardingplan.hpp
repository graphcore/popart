// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SHARDINGPLAN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SHARDINGPLAN_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <popart/names.hpp>
#include <popart/op.hpp>

#include "popart/tensorinfo.hpp"

namespace popart {
class Graph;

/**
 * Enum type that specifies how an Op should be sharded.
 */
enum class ShardingMethod {
  /// Use DynamicSlice/DynamicUpdate Ops to split and concatenate input/output
  /// tensors when sharding the Op.
  /// The Op will be unrolled on the sharded dimension.
  DynamicShard = 0,
  /// Use Slice/Concat Ops to split and concatenate input/output
  /// tensors when sharding the Op.
  /// The Op will be unrolled on the sharded dimension.
  StaticShard,
  /// Shard the Op by replacing it with a LoopOp that iterates over the sharded
  /// dimension. Implies using DynamicShard before, after and inside the LoopOp.
  Loop
};

/**
 * Enum type that specifies how a tensor should be sharded
 */
enum class ShardTensorType {
  /// Shard tensors are derived by slicing the tensor
  Slice = 0,
  /// Shard tensors are derived by adding an offset to the tensor
  Offset
};

/**
 * Map from input to sharded tensor IDs.
 * Key:   TensorId connected to the Op before sharding
 * Value: TensorIds to be connected to the sharded Ops after sharding
 */
using ShardIdMap = std::map<TensorId, std::vector<TensorId>>;

/**
 * Class that describes the sharded tensors
 */
class ShardTensorInfo {
public:
  ShardTensorInfo() : id(), info(), infos(), type() {}

  /// Construct ShardTensorInfo from parameters.
  /// \param id_ The method of sharding the Op
  /// \param info_ The sharded TensorIds to be connected
  /// \param infos_ The graph which contains the TensorIds
  ShardTensorInfo(TensorId id_,
                  TensorInfo info_,
                  std::vector<TensorInfo> infos_)
      : id(id_), info(info_), infos(infos_), type() {}

  /// Construct ShardTensorInfo from parameters.
  /// \param id_ The method of sharding the Op
  /// \param info_ The sharded TensorIds to be connected
  /// \param infos_ The graph which contains the TensorIds
  /// \param type_ The settings to apply onto the sharded ops
  ShardTensorInfo(TensorId id_,
                  TensorInfo info_,
                  std::vector<TensorInfo> infos_,
                  ShardTensorType type_)
      : id(id_), info(info_), infos(infos_), type(type_) {}

  /// TensorId to be connected as a replacement after sharding
  TensorId id;
  /// TensorInfo of the tensor to be connected as a replacement
  TensorInfo info;
  /// TensorInfos describing how to shard the TensorId before connecting
  /// to the sharded Ops
  std::vector<TensorInfo> infos;
  /// Type of sharded tensor (sliceable or offsetable are currently supported)
  ShardTensorType type;
};

/**
 * Map from input to sharded tensor infos
 * Key:   TensorId connected to the Op before sharding
 * Value: ShardTensorInfo
 */
using ShardInfoMap = std::map<TensorId, ShardTensorInfo>;

/**
 * Class that describes what Op::Settings to apply after sharding an Op
 *
 * <pre>
 *            Init           < getPreSetting() will be applied
 *             |
 *  Op ---- DynamicUpdate    < getShardSettings().at(0) will be applied
 *             |
 *  Op ---- DynamicUpdate    < getShardSettings().at(1) will be applied
 *             |
 *  Op ---- DynamicUpdate    < getShardSettings().at(2) will be applied
 *             |
 *           IdLossOp        < getPostSetting() will be applied
 * </pre>
 *
 * If not specified, the pre-sharding Op settings will be used instead.
 */
class ShardOpSettings {
public:
  Op::Settings getPreSetting() const { return preSettings.at(0); }
  Op::Settings getPostSetting() const { return postSettings.at(0); }
  bool hasPreSetting() const { return preSettings.size(); }
  bool hasPostSetting() const { return postSettings.size(); }
  const std::vector<Op::Settings> &getShardSettings() const {
    return shardSettings;
  }
  void setPreSettings(Op::Settings setting) { preSettings = {setting}; }
  void setShardSettings(std::vector<Op::Settings> setting) {
    shardSettings = setting;
  }
  void setPostSettings(Op::Settings setting) { postSettings = {setting}; }

private:
  std::vector<Op::Settings> preSettings;
  std::vector<Op::Settings> shardSettings;
  std::vector<Op::Settings> postSettings;
};

/**
 * Class that describes how an Op should be sharded
 */
class ShardingPlan {
public:
  /// Construct a ShardingPlan from parameters.
  /// \param method The method of sharding the Op
  /// \param settings The settings to apply onto the sharded ops
  ShardingPlan(ShardingMethod method,
               ShardOpSettings settings = ShardOpSettings());

  /// Construct a ShardingPlan from parameters.
  /// \param method The method of sharding the Op
  /// \param shardMap The sharded TensorIds to be connected
  /// \param graph The graph which contains the TensorIds
  /// \param settings The settings to apply onto the sharded ops
  ShardingPlan(ShardingMethod method,
               const ShardIdMap &shardMap,
               Graph &graph,
               ShardOpSettings settings = ShardOpSettings());

  /// Construct a ShardingPlan from parameters.
  /// \param method The method of sharding the Op
  /// \param shardMap The sharded TensorInfos to be connected
  /// \param settings The settings to apply onto the sharded ops
  ShardingPlan(ShardingMethod method,
               const ShardInfoMap &shardMap,
               ShardOpSettings settings = ShardOpSettings());

  bool canDynamicShard() const;
  bool canLoop() const;

  ShardingMethod getMethod() const { return method; }

  ShardIdMap getIdMap() const { return shardIdMap; }

  ShardInfoMap getInfoMap() const { return shardInfoMap; }

  void insertIdMap(const ShardIdMap &shardMap_, Graph &graph_);

  void insertInfoMap(const ShardInfoMap &shardMap_);

  const ShardOpSettings &getOpSettings() const { return shardSettings; }

  void setOpSettings(ShardOpSettings shardSettings_) {
    shardSettings = shardSettings_;
  }

  int64_t getTotalNumShards() const { return totalNumShards; }

  void setTotalNumShards(int64_t num) { totalNumShards = num; }

private:
  void fillInfoMapFromIdMap(TensorId id, Graph &graph);

  ShardingMethod method;
  ShardIdMap shardIdMap;
  ShardInfoMap shardInfoMap;
  ShardOpSettings shardSettings;
  int64_t totalNumShards;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_SHARDINGPLAN_HPP_
