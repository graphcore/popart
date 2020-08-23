// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORLOCATION_HPP
#define GUARD_NEURALNET_TENSORLOCATION_HPP

#include <memory>
#include <vector>

#include <popart/names.hpp>

namespace popart {

enum class TensorStorage { Undefined = 0, OnChip = 1, OffChip = 2 };

enum class TileSet { Compute = 0, IO = 1 };
using VGraphIdAndIoTile = std::pair<VGraphId, TileSet>;

enum class ReplicatedTensorSharding { Off = 0, On = 1 };

class TensorLocation {
public:
  TensorLocation();
  TensorLocation(TensorStorage storage_);
  TensorLocation(std::vector<int64_t> serialized);
  TensorLocation(TensorStorage storage_,
                 TileSet loadTileSet_,
                 TileSet storageTileSet_,
                 ReplicatedTensorSharding replicatedTensorSharding_);
  TensorLocation(TensorStorage storage_,
                 ReplicatedTensorSharding replicatedTensorSharding_);

  TensorLocation &operator=(const TensorLocation &rhs) = default;
  bool operator==(const TensorLocation &rhs);
  bool operator!=(const TensorLocation &rhs);

  std::vector<int64_t> serialize() const;

  bool isRemote() const;

  // Permanent tensor storage: OnChip or OffChip
  TensorStorage storage;
  // Load tensor through IO tiles
  TileSet loadTileSet;
  // If OnChip: Store on IO tiles
  // (relevant for replicated tensor sharded tensors)
  TileSet storageTileSet;
  // Enable replicated tensor sharding
  // (relevant for weights and optimizer states)
  ReplicatedTensorSharding replicatedTensorSharding;
};

bool isValidTensorLocation(const TensorLocation tensorLocation);

std::ostream &operator<<(std::ostream &, const TensorStorage &);
std::ostream &operator<<(std::ostream &, const TileSet &);
std::ostream &operator<<(std::ostream &, const ReplicatedTensorSharding &);
std::ostream &operator<<(std::ostream &, const TensorLocation &);

} // namespace popart

#endif
