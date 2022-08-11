// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include <ostream>
#include <vector>
#include <popart/commgroup.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/variablesettings.hpp>

#include "popart/logging.hpp"

namespace popart {

std::ostream &operator<<(std::ostream &os, const VariableRetrievalMode &vrm) {
  switch (vrm) {
  case VariableRetrievalMode::OnePerGroup:
    os << "OnePerGroup";
    break;
  case VariableRetrievalMode::AllReduceReplicas:
    os << "AllReduceReplicas";
    break;
  case VariableRetrievalMode::AllReplicas:
    os << "AllReplicas";
    break;
  default:
    os << "Undefined VariableRetrievalMode: " << static_cast<int>(vrm);
    break;
  }
  return os;
}

/// "Default" constructor, defaults CommGroup to [All, 0] and retrievalMode to
/// OnePerGroup
VariableSettings::VariableSettings()
    : sharedVariableDomain(CommGroup(CommGroupType::All, 0)),
      retrievalMode(VariableRetrievalMode::OnePerGroup) {}

/// Defaults VariableRetrievalMode to OnePerGroup
VariableSettings::VariableSettings(CommGroup sharedVariableDomain_)
    : sharedVariableDomain(sharedVariableDomain_),
      retrievalMode(VariableRetrievalMode::OnePerGroup) {
  verify();
}

/// Defaults CommGroup to [All, 0]
VariableSettings::VariableSettings(VariableRetrievalMode retrievalMode_)
    : sharedVariableDomain(CommGroup(CommGroupType::All, 0)),
      retrievalMode(retrievalMode_) {
  verify();
}

/// Entirely custom VariableSettings
VariableSettings::VariableSettings(CommGroup sharedVariableDomain_,
                                   VariableRetrievalMode retrievalMode_)
    : sharedVariableDomain(sharedVariableDomain_),
      retrievalMode(retrievalMode_) {
  verify();
}

unsigned
VariableSettings::numReplicasReturningVariable(unsigned replicaCount) const {

  // If instruction is to return from all,
  // replicas are ungrouped, or if group size
  // is one, we know to return for all replicas

  if (retrievalMode == VariableRetrievalMode::AllReplicas ||
      sharedVariableDomain.type == CommGroupType::None ||
      sharedVariableDomain.replicaGroupSize == 1)
    return replicaCount;

  // Logical Point: Beyond this we know retrievalMode is
  // OnePerGroup or AllReduceReplicas, meaning we will return
  //   at most one per domain/group.

  // Make sure the dimension of the commGroup is a valid number.
  if (sharedVariableDomain.type != CommGroupType::All &&
      sharedVariableDomain.replicaGroupSize < 1) {
    throw internal_error("Attempting to use 0 as a CommGroupSize, "
                         "groups of type {} must have legal size.\n",
                         sharedVariableDomain.type);
  }

  // differentiate between options returning one and group-size.
  // Also
  switch (sharedVariableDomain.type) {
  case CommGroupType::All:
    return 1;
  case CommGroupType::Consecutive:
  case CommGroupType::Orthogonal:
    return replicaCount / sharedVariableDomain.replicaGroupSize;
  case CommGroupType::None:
    logging::err(
        "Unreachable point, logic should have circumvented this case.\n");
    break;
  case CommGroupType::N:
  default:
    logging::err("Bad CommGroupType {} in VariableSetting.\n",
                 sharedVariableDomain.type);
    break;
  }

  logging::err("Unresolved switch-case, logic should have circumvented it.\n");
  return -1;
}

unsigned VariableSettings::getGroupCount(unsigned replicaCount) const {
  if (sharedVariableDomain.type == CommGroupType::None ||
      sharedVariableDomain.replicaGroupSize == 1) {
    return replicaCount;
  }

  switch (sharedVariableDomain.type) {
  case CommGroupType::All:
    return 1;
  case CommGroupType::None:
    return replicaCount;
  case CommGroupType::Consecutive:
  case CommGroupType::Orthogonal:
    if (!sharedVariableDomain.replicaGroupSize) {
      throw internal_error("Attempting to use 0 as a CommGroupSize, "
                           "groups of type {} must have legal size.\n",
                           sharedVariableDomain.type);
    }
    return replicaCount / sharedVariableDomain.replicaGroupSize;
  case CommGroupType::N:
  default:
    throw internal_error("Bad CommGroupType {} in VariableSetting.\n",
                         sharedVariableDomain.type);
  }
}

unsigned VariableSettings::getStride(unsigned replicaCount) const {
  switch (sharedVariableDomain.type) {
  case CommGroupType::All:
  case CommGroupType::None:
  case CommGroupType::Consecutive:
    return 1;
  case CommGroupType::Orthogonal:
    if (!sharedVariableDomain.replicaGroupSize) {
      throw internal_error("Attempting to use 0 as a CommGroupSize, "
                           "groups of type {} must have legal size.\n",
                           sharedVariableDomain.type);
    }
    return replicaCount / sharedVariableDomain.replicaGroupSize;
  case CommGroupType::N:
  default:
    throw internal_error("Bad CommGroupType {} in VariableSetting.\n",
                         sharedVariableDomain.type);
  }
}

unsigned VariableSettings::getRealGroupSize(unsigned replicaCount) const {
  switch (sharedVariableDomain.type) {
  case CommGroupType::All:
    return replicaCount;
  case CommGroupType::None:
    return 1;
  case CommGroupType::Consecutive:
  case CommGroupType::Orthogonal:
    return sharedVariableDomain.replicaGroupSize;
  default:
    throw internal_error("Bad CommGroupType {} in VariableSetting.\n",
                         sharedVariableDomain.type);
  }
}

unsigned VariableSettings::getGroupRepresentative(unsigned group) const {
  switch (sharedVariableDomain.type) {
  case CommGroupType::All:
    return 0;

  case CommGroupType::Consecutive:
    return sharedVariableDomain.replicaGroupSize * group;

  case CommGroupType::Orthogonal:
  case CommGroupType::None:
    return group;

  default:
    logging::err("Bad CommGroupType {} in VariableSetting.\n",
                 sharedVariableDomain.type);
    return -1;
  }
}

void VariableSettings::verify() {
  int throw_error = 0;
  auto type       = static_cast<int64_t>(sharedVariableDomain.type);

  if (type < 0 || type >= static_cast<int64_t>(CommGroupType::N)) {
    throw_error++;
    logging::err("Bad Commgroup: ", sharedVariableDomain.type);
  }

  if (sharedVariableDomain.type != CommGroupType::All &&
      sharedVariableDomain.type != CommGroupType::None &&
      sharedVariableDomain.replicaGroupSize < 1) {
    throw_error++;
    logging::err("Bad ReplicaGroupSize ({}) for domain type ({})!",
                 sharedVariableDomain.replicaGroupSize,
                 sharedVariableDomain.replicaGroupSize);
  }

  if (retrievalMode != VariableRetrievalMode::OnePerGroup &&
      retrievalMode != VariableRetrievalMode::AllReduceReplicas &&
      retrievalMode != VariableRetrievalMode::AllReplicas) {
    throw_error++;
    logging::err("Bad VariableRetrievalMode: {}", retrievalMode);
  }

  if (throw_error) {
    throw internal_error("VariableSettings had {} errors!", throw_error);
  }
}

Shape VariableSettings::shapeOnReplica(const Shape full_shape,
                                       unsigned replicaCount,
                                       const TensorId name) const {
  auto numGroups = getGroupCount(replicaCount);

  if (numGroups == 1) {
    return Shape(full_shape);
  }
  if (numGroups != full_shape[0]) {
    throw internal_error("Return mismatch with possibly appended "
                         "outer dimension ({}) of Tensor: \"{}\". "
                         "should match numGroups: {}",
                         full_shape[0],
                         name,
                         numGroups);
  }

  Shape reshape;
  for (int i = 1; i < full_shape.size(); i++) {
    reshape.push_back(full_shape[i]);
  }

  return reshape;
}

Shape VariableSettings::shapeOnHost(const Shape replicaShape,
                                    unsigned replicaCount) const {
  auto numGroups = getGroupCount(replicaCount);

  // If there's only one group, the replica and host dimensions are the same
  if (numGroups == 1) {
    return Shape(replicaShape);
  }

  // Otherwise prepend numGroups to the shape
  Shape hostShape = {numGroups};
  hostShape.insert(hostShape.end(), replicaShape.begin(), replicaShape.end());
  return hostShape;
}

std::vector<std::vector<std::int64_t>>
VariableSettings::groups(unsigned replicaCount) const {
  std::vector<std::vector<std::int64_t>> groups;
  std::vector<std::int64_t> group;
  if (sharedVariableDomain.type == CommGroupType::All) {
    for (auto i = 0; i < replicaCount; i++) {
      group.push_back(i);
    }
    groups.push_back(group);
    return groups;
  }
  if (sharedVariableDomain.type == CommGroupType::None) {
    for (auto i = 0; i < replicaCount; i++) {
      group = std::vector<std::int64_t>();
      group.push_back(i);
      groups.push_back(group);
    }
    return groups;
  }
  auto totalGroups = getGroupCount(replicaCount);
  auto groupInc    = getStride(replicaCount);
  auto groupSize   = getRealGroupSize(replicaCount);

  for (auto groupIdx = 0; groupIdx < totalGroups; groupIdx++) {
    group      = std::vector<std::int64_t>();
    auto start = getGroupRepresentative(groupIdx);
    auto end   = start + (groupInc * groupSize);
    for (auto repId = start; repId < end; repId += groupInc) {
      group.push_back(repId);
    }
    groups.push_back(group);
  }
  return groups;
}

bool VariableSettings::operator==(VariableSettings other) {
  return other.getRetrievalMode() == retrievalMode &&
         other.getSharedVariableDomain().type == sharedVariableDomain.type &&
         other.getSharedVariableDomain().replicaGroupSize ==
             sharedVariableDomain.replicaGroupSize;
}

bool VariableSettings::operator!=(VariableSettings other) {
  return !this->operator==(other);
}

std::ostream &operator<<(std::ostream &os, const VariableSettings &vs) {
  return os << "VariableSettings: [" << vs.getSharedVariableDomain() << ", "
            << vs.getRetrievalMode() << "]";
}
} // namespace popart
