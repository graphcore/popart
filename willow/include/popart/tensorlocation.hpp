// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORLOCATION_HPP
#define GUARD_NEURALNET_TENSORLOCATION_HPP

#include <cstdint>
#include <iosfwd>
#include <set>
#include <utility>
#include <vector>
#include <popart/commgroup.hpp>
#include <popart/names.hpp>

namespace popart {

/**
 * Enum type that determines where a tensor is stored.
 */
enum class TensorStorage {
  /// Store the tensor in on-chip memory.
  OnChip = 0,
  /// Store the tensor in streaming memory.
  OffChip = 1,
  /// Number of values.
  N = 2
};

/**
 * Enum type to specify a set of tiles.
 */
enum class TileSet {
  /// The set of tiles designated for compute operations.
  Compute = 0,
  /// The set of tiles designated for IO operations.
  IO = 1,
  /// Undefined (no) tile set.
  Undefined = 2,
  /// Number of values.
  N = 3
};

using VGraphIdAndTileSet = std::pair<VGraphId, TileSet>;

struct VGraphIdAndTileSetCmp {
  bool operator()(VGraphIdAndTileSet const &a,
                  VGraphIdAndTileSet const &b) const {
    if (a.first != unusedVGraphId &&
        (a.first < b.first || b.first == unusedVGraphId)) {
      return true;
    }
    if (b.first != unusedVGraphId &&
        (b.first < a.first || a.first == unusedVGraphId)) {
      return false;
    }
    if (a.second < b.second) {
      return true;
    } else {
      return false;
    }
  }
};

using VGraphIdAndTileSetSet =
    std::set<VGraphIdAndTileSet, VGraphIdAndTileSetCmp>;

/**
 * Enum type to specify whether to shard tensors over replicas.
 */
enum class ReplicatedTensorSharding {
  /// Don't shard tensors over replicas.
  Off = 0,
  /// Do shard tensors over replicas.
  On = 1,
  /// Number of values
  N = 2
};

/**
 * Class that describes the memory characteristics of one or multiple tensors.
 *
 * See also: SessionOptions.
 */
class TensorLocation {
public:
  /// Equivalent to calling
  /// TensorLocation(TensorStorage::Undefined,
  ///                TileSet::Compute,
  ///                TileSet::Compute,
  ///                ReplicatedTensorSharding::Off)
  TensorLocation();
  /// Equivalent to calling
  /// TensorLocation(storage,
  ///                TileSet::Compute,
  ///                TileSet::Compute,
  ///                ReplicatedTensorSharding::Off)
  TensorLocation(TensorStorage storage);

  /// Equivalent to calling
  /// TensorLocation(storage,
  ///                TileSet::Compute,
  ///                TileSet::Compute,
  ///                replicatedTensorSharding)
  TensorLocation(TensorStorage storage,
                 ReplicatedTensorSharding replicatedTensorSharding);

  /// Equivalent to calling
  /// TensorLocation(storage,
  ///                TileSet::Compute,
  ///                TileSet::Compute,
  ///                replicatedTensorSharding,
  ///                shardingDomain)
  TensorLocation(TensorStorage storage,
                 ReplicatedTensorSharding replicatedTensorSharding,
                 CommGroup shardingDomain);

  /// Construct a TensorLocation from parameters.
  /// \param storage The memory location of the tensor(s).
  /// \param loadTileSet The tiles through which the tensor(s) are loaded onto
  ///    the chip.
  /// \param storageTileSet The tiles on which the tensor(s) are stored.
  /// \param replicatedTensorSharding Whether to apply replicated tensor.
  ///    sharding.
  TensorLocation(TensorStorage storage,
                 TileSet loadTileSet,
                 TileSet storageTileSet,
                 ReplicatedTensorSharding replicatedTensorSharding);

  /// Construct a TensorLocation from parameters.
  /// \param storage The memory location of the tensor(s).
  /// \param loadTileSet The tiles through which the tensor(s) are loaded onto
  ///                    the chip.
  /// \param storageTileSet The tiles on which the tensor(s) are stored.
  /// \param replicatedTensorSharding Whether to apply replicated tensor.
  ///                                 sharding.
  /// \param shardingDomain GCL communication group across which to shard
  ///                       the tensor. Perpendicular replicas will not shard,
  ///                       and reduce gradients normally (via AllReduce).
  ///                       Defaults to sharding across all replicas.
  TensorLocation(TensorStorage storage,
                 TileSet loadTileSet,
                 TileSet storageTileSet,
                 ReplicatedTensorSharding replicatedTensorSharding,
                 CommGroup shardingDomain);

  // Construct a TensorsorLocation from a previously serialised instance
  // (not currently part of public API).
  TensorLocation(std::vector<int64_t> serialized);

  // Equality operator for TensorLocation
  // (not currently part of public API).
  bool operator==(const TensorLocation &rhs) const;
  // Inequality operator for TensorLocation
  // (not currently part of public API).
  bool operator!=(const TensorLocation &rhs) const;

  // Function that serialises a TensorLocation to a list of int64_t values
  // (not currently part of public API).
  std::vector<int64_t> serialize() const;

  // Returns `true` when the location is either off-chip or remote tensor
  // sharding (not currently part of public API).
  bool isRemote() const;

  /// The memory location of the tensor(s).
  TensorStorage storage;
  /// The tiles through which the tensor(s) are loaded onto
  /// the chip.
  TileSet loadTileSet;
  /// The tiles on which the tensor(s) are stored.
  TileSet storageTileSet;
  /// Whether to apply replicated tensor sharding (RTS) or not.
  ReplicatedTensorSharding replicatedTensorSharding;
  /// The GCL comm groups across which to shard the tensor
  CommGroup shardingDomain;
};

std::ostream &operator<<(std::ostream &, const VGraphIdAndTileSet &);
std::ostream &operator<<(std::ostream &, const VGraphIdAndTileSetSet &);
std::ostream &operator<<(std::ostream &, const TensorStorage &);
std::ostream &operator<<(std::ostream &, const TileSet &);
std::ostream &operator<<(std::ostream &, const ReplicatedTensorSharding &);
std::ostream &operator<<(std::ostream &, const TensorLocation &);

} // namespace popart

#endif
