// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORLOCATION_HPP
#define GUARD_NEURALNET_TENSORLOCATION_HPP

#include <memory>
#include <vector>

#include <popart/names.hpp>

namespace popart {

/**
 * Enum type that determines where a tensor is stored.
 */
enum class TensorStorage {
  /// Location unspecified.
  Undefined = 0,
  /// Store the tensor in on-chip memory.
  OnChip = 1,
  /// Store the tensor in streaming memory.
  OffChip = 2
};

/**
 * Enum type to specify a set of tiles.
 */
enum class TileSet {
  /// The set of tiles designated for compute operations.
  Compute = 0,
  /// The set of tiles designated for IO operations.
  IO = 1
};

using VGraphIdAndTileSet = std::pair<VGraphId, TileSet>;

/**
 * Enum type to specify whether to shard tensors over replicas.
 */
enum class ReplicatedTensorSharding {
  /// Don't shard tensors over replicas.
  Off = 0,
  /// Do shard tensors over replicas.
  On = 1
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
  /// Construct a TensorLocation from parameters.
  /// \param storage the memory location of the tensor(s).
  /// \param loadTileSet the tiles through which the tensor(s) are loaded onto
  ///    the chip.
  /// \param storageTileSet the tiles on which the tensor(s) are stored.
  /// \param replicatedTensorSharding whether to apply replicated tensor
  ///    sharding.
  TensorLocation(TensorStorage storage,
                 TileSet loadTileSet,
                 TileSet storageTileSet,
                 ReplicatedTensorSharding replicatedTensorSharding);

  // Construct a TensorsorLocation from a previously serialised instance
  // (not currently part of public API).
  TensorLocation(std::vector<int64_t> serialized);

  // Assignment operator for TensorLocation
  // (not currently part of public API).
  TensorLocation &operator=(const TensorLocation &rhs) = default;
  // Equality operator for TensorLocation
  // (not currently part of public API).
  bool operator==(const TensorLocation &rhs);
  // Inequality operator for TensorLocation
  // (not currently part of public API).
  bool operator!=(const TensorLocation &rhs);

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
};

bool isValidTensorLocation(const TensorLocation tensorLocation);

std::ostream &operator<<(std::ostream &, const TensorStorage &);
std::ostream &operator<<(std::ostream &, const TileSet &);
std::ostream &operator<<(std::ostream &, const ReplicatedTensorSharding &);
std::ostream &operator<<(std::ostream &, const TensorLocation &);

} // namespace popart

#endif
