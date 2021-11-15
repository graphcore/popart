// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/commgroup.hpp>
#include <popart/error.hpp>

namespace popart {

CommGroup::CommGroup() = default;

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
