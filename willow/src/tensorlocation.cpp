// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/error.hpp>
#include <popart/tensorlocation.hpp>

namespace popart {

TensorLocation::TensorLocation()
    : storage(TensorStorage::OnChip), loadTileSet(TileSet::Compute),
      storageTileSet(TileSet::Compute),
      replicatedTensorSharding(ReplicatedTensorSharding::Off) {}

TensorLocation::TensorLocation(TensorStorage storage_)
    : storage(storage_), loadTileSet(TileSet::Compute),
      storageTileSet(TileSet::Compute),
      replicatedTensorSharding(ReplicatedTensorSharding::Off) {}

TensorLocation::TensorLocation(std::vector<int64_t> serialized)
    : storage(static_cast<TensorStorage>(serialized[0])),
      loadTileSet(static_cast<TileSet>(serialized[1])),
      storageTileSet(static_cast<TileSet>(serialized[2])),
      replicatedTensorSharding(
          static_cast<ReplicatedTensorSharding>(serialized[3])) {}

TensorLocation::TensorLocation(
    TensorStorage storage_,
    ReplicatedTensorSharding replicatedTensorSharding_)
    : storage(storage_), loadTileSet(TileSet::Compute),
      storageTileSet(TileSet::Compute),
      replicatedTensorSharding(replicatedTensorSharding_) {}

TensorLocation::TensorLocation(
    TensorStorage storage_,
    TileSet loadTileSet_,
    TileSet storageTileSet_,
    ReplicatedTensorSharding replicatedTensorSharding_)
    : storage(storage_), loadTileSet(loadTileSet_),
      storageTileSet(storageTileSet_),
      replicatedTensorSharding(replicatedTensorSharding_) {}

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
          static_cast<int64_t>(replicatedTensorSharding)};
}

bool TensorLocation::isRemote() const {
  return (replicatedTensorSharding == ReplicatedTensorSharding::On ||
          (storage == TensorStorage::OffChip));
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
  ost << ")";
  return ost;
}

} // namespace popart
