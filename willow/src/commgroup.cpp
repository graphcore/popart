// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include <ostream>
#include <type_traits>
#include <popart/commgroup.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/replicagrouping.hpp"

namespace popart {

CommGroup::CommGroup() = default;

CommGroup::CommGroup(const ReplicaGrouping &grouping) {
  const auto &numReplicas = grouping.getNumReplicas();
  const auto &stride      = grouping.getStride();
  const auto &groupSize   = grouping.getGroupSize();
  const auto &numGroups   = grouping.getNumGroups();

  if (groupSize == 1) {
    type             = CommGroupType::None;
    replicaGroupSize = 0;
  } else if (groupSize == numReplicas) {
    type             = CommGroupType::All;
    replicaGroupSize = 0;
  } else if (stride == numGroups) {
    type             = CommGroupType::Orthogonal;
    replicaGroupSize = groupSize;
  } else if (stride == 1) {
    type             = CommGroupType::Consecutive;
    replicaGroupSize = groupSize;
  } else {
    throw popart::error(
        "The '{}' cannot be converted to a `popart::CommGroup`.",
        grouping.str());
  }
}

ReplicaGrouping CommGroup::toReplicaGrouping(unsigned numReplicas) const {
  unsigned stride    = 0;
  unsigned groupSize = 0;

  switch (type) {
  case CommGroupType::All:
    stride    = 1;
    groupSize = numReplicas;
    break;
  case CommGroupType::Consecutive:
    stride    = 1;
    groupSize = replicaGroupSize;
    break;
  case CommGroupType::Orthogonal:
    stride    = numReplicas / replicaGroupSize;
    groupSize = replicaGroupSize;
    break;
  case CommGroupType::None:
    stride    = 1;
    groupSize = 1;
    break;
  default:
    throw error("Invalid type '[]' for a CommGroup.", type);
  }

  return ReplicaGrouping(numReplicas, stride, groupSize);
}

bool CommGroup::operator==(const CommGroup &other) const {
  return type == other.type && replicaGroupSize == other.replicaGroupSize;
}
bool CommGroup::operator!=(const CommGroup &other) const {
  return !(*this == other);
}

std::ostream &operator<<(std::ostream &os, CommGroupType commType) {
  switch (commType) {
  case CommGroupType::All:
    os << "All";
    break;
  case CommGroupType::Consecutive:
    os << "Consecutive";
    break;
  case CommGroupType::Orthogonal:
    os << "Orthogonal";
    break;
  case CommGroupType::None:
    os << "None";
    break;
  default:
    os << "Invalid CommGroup " << static_cast<int64_t>(commType);
    logging::err("Unsupported CommGroupType {}",
                 std::underlying_type_t<CommGroupType>(commType));
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const CommGroup &group) {
  os << "CommGroup(type=" << group.type
     << ", replicaGroupSize=" << group.replicaGroupSize << ")";
  return os;
}

} // namespace popart
