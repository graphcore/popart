// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>
#include <popart/commgroup.hpp>
#include <popart/error.hpp>
#include <popart/tensorlocation.hpp>

#include "popart/logging.hpp"

namespace popart {

TensorLocation::TensorLocation()
    : storage(TensorStorage::OnChip), loadTileSet(TileSet::Compute),
      storageTileSet(TileSet::Compute),
      replicatedTensorSharding(ReplicatedTensorSharding::Off),
      shardingDomain() {}

TensorLocation::TensorLocation(TensorStorage storage_)
    : storage(storage_), loadTileSet(TileSet::Compute),
      storageTileSet(TileSet::Compute),
      replicatedTensorSharding(ReplicatedTensorSharding::Off),
      shardingDomain() {}

TensorLocation::TensorLocation(std::vector<int64_t> serialized)
    : storage(static_cast<TensorStorage>(serialized[0])),
      loadTileSet(static_cast<TileSet>(serialized[1])),
      storageTileSet(static_cast<TileSet>(serialized[2])),
      replicatedTensorSharding(
          static_cast<ReplicatedTensorSharding>(serialized[3])),
      shardingDomain(static_cast<CommGroupType>(serialized[4]),
                     static_cast<unsigned>(serialized[5])) {}

TensorLocation::TensorLocation(
    TensorStorage storage_,
    ReplicatedTensorSharding replicatedTensorSharding_)
    : storage(storage_), loadTileSet(TileSet::Compute),
      storageTileSet(TileSet::Compute),
      replicatedTensorSharding(replicatedTensorSharding_), shardingDomain() {}

TensorLocation::TensorLocation(
    TensorStorage storage_,
    ReplicatedTensorSharding replicatedTensorSharding_,
    CommGroup shardingDomain_)
    : storage(storage_), loadTileSet(TileSet::Compute),
      storageTileSet(TileSet::Compute),
      replicatedTensorSharding(replicatedTensorSharding_),
      shardingDomain(shardingDomain_) {}

TensorLocation::TensorLocation(
    TensorStorage storage_,
    TileSet loadTileSet_,
    TileSet storageTileSet_,
    ReplicatedTensorSharding replicatedTensorSharding_)
    : storage(storage_), loadTileSet(loadTileSet_),
      storageTileSet(storageTileSet_),
      replicatedTensorSharding(replicatedTensorSharding_) {}

TensorLocation::TensorLocation(
    TensorStorage storage_,
    TileSet loadTileSet_,
    TileSet storageTileSet_,
    ReplicatedTensorSharding replicatedTensorSharding_,
    CommGroup shardingDomain_)
    : storage(storage_), loadTileSet(loadTileSet_),
      storageTileSet(storageTileSet_),
      replicatedTensorSharding(replicatedTensorSharding_),
      shardingDomain(shardingDomain_) {}

bool TensorLocation::operator==(const TensorLocation &rhs) const {
  return serialize() == rhs.serialize();
}

bool TensorLocation::operator!=(const TensorLocation &rhs) const {
  return serialize() != rhs.serialize();
}

std::vector<int64_t> TensorLocation::serialize() const {
  return {static_cast<int64_t>(storage),
          static_cast<int64_t>(loadTileSet),
          static_cast<int64_t>(storageTileSet),
          static_cast<int64_t>(replicatedTensorSharding),
          static_cast<int64_t>(shardingDomain.type),
          static_cast<int64_t>(shardingDomain.replicaGroupSize)};
}

bool TensorLocation::isRemote() const {
  return (replicatedTensorSharding == ReplicatedTensorSharding::On ||
          (storage == TensorStorage::OffChip));
}

std::ostream &operator<<(std::ostream &ost, const VGraphIdAndTileSet &vgidats) {
  ost << "(";
  ost << vgidats.first << ", " << vgidats.second;
  ost << ")";
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const VGraphIdAndTileSetSet &set) {
  ost << "[";
  ost << logging::join(set.begin(), set.end(), ", ");
  ost << "]";
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const TensorStorage &ts) {
  switch (ts) {
  case (TensorStorage::OnChip): {
    ost << "OnChip";
    break;
  }
  case (TensorStorage::OffChip): {
    ost << "OffChip";
    break;
  }
  default: {
    throw error("Unexpected value for TensorStorage {}", static_cast<int>(ts));
  }
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const TileSet &ts) {
  switch (ts) {
  case (TileSet::Compute): {
    ost << "Compute";
    break;
  }
  case (TileSet::IO): {
    ost << "IO";
    break;
  }
  case (TileSet::Undefined): {
    ost << "Undefined";
    break;
  }
  default: {
    throw error("Unexpected value for TileSet {}", static_cast<int>(ts));
  }
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost,
                         const ReplicatedTensorSharding &rts) {
  switch (rts) {
  case (ReplicatedTensorSharding::Off): {
    ost << "Off";
    break;
  }
  case (ReplicatedTensorSharding::On): {
    ost << "On";
    break;
  }
  default: {
    throw error("Unexpected value for ReplicatedTensorSharding {}",
                static_cast<int>(rts));
  }
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const TensorLocation &tl) {
  ost << "(";
  ost << tl.storage;
  ost << ", loadTileSet=" << tl.loadTileSet;
  ost << ", storageTileSet=" << tl.storageTileSet;
  ost << ", RTS=" << tl.replicatedTensorSharding;
  if (tl.replicatedTensorSharding == ReplicatedTensorSharding::On) {
    ost << ", shardingDomain=" << tl.shardingDomain;
  }
  ost << ")";
  return ost;
}

} // namespace popart
