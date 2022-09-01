// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>
#include <popart/commgroup.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/variablesettings.hpp>

#include "popart/logging.hpp"
#include "popart/replicagrouping.hpp"
#include "variablesettingsdomain.hpp"

namespace popart {

/**
 * Get the number of replicas returning a variable. Used internally in
 * `popart::VariableSettings`, when the `popart::VariableSettings::domain_` is a
 * `popart::ReplicaGrouping`.
 */
static unsigned
numReplicasReturningVariable(const ReplicaGrouping &grouping,
                             const VariableRetrievalMode &mode) {
  switch (mode) {
  case VariableRetrievalMode::OnePerGroup:
  case VariableRetrievalMode::AllReduceReplicas:
    return grouping.getNumGroups();
    break;
  case VariableRetrievalMode::AllReplicas:
    return grouping.getNumReplicas();
    break;
  default:
    throw internal_error("An invalid retrieval mode '{}' was encountered in a "
                         "`VariableSettings` instance.",
                         mode);
    break;
  }
}

/**
 * Get the group count. Used internally in `popart::VariableSettings`, when the
 * `popart::VariableSettings::domain_` is a `popart::ReplicaGrouping`.
 */
static unsigned getGroupCount(const ReplicaGrouping &grouping) {
  return grouping.getNumGroups();
}

/**
 * Get the stride. Used internally in `popart::VariableSettings`, when the
 * `popart::VariableSettings::domain_` is a `popart::ReplicaGrouping`.
 */
static unsigned getStride(const ReplicaGrouping &grouping) {
  return grouping.getStride();
}

/**
 * Get the group size. Used internally in `popart::VariableSettings`, when the
 * `popart::VariableSettings::domain_` is a `popart::ReplicaGrouping`.
 */
static unsigned getRealGroupSize(const ReplicaGrouping &grouping) {
  return grouping.getGroupSize();
}

/**
 * Get the group representative. Used internally in `popart::VariableSettings`,
 * when the `popart::VariableSettings::domain_` is a `popart::ReplicaGrouping`.
 */
static unsigned getGroupRepresentative(const ReplicaGrouping &grouping,
                                       unsigned group) {
  return grouping.getReplicaAt(group);
}

/**
 * Get the shape on replica. Used internally in `popart::VariableSettings`, when
 * the `popart::VariableSettings::domain_` is a `popart::ReplicaGrouping`.
 */
static Shape shapeOnReplica(const ReplicaGrouping &grouping,
                            const Shape &shape,
                            const TensorId &name) {
  auto numGroups = grouping.getNumGroups();

  if (numGroups == 1) {
    return shape;
  }
  POPART_CHECK_GT(shape.size(), 0)
      << "Tensor '" << name
      << "' should have at least one dimension when used with "
         "`popart::VariableSettings` with more than 1 replica groups.";
  POPART_CHECK_EQ(numGroups, shape[0])
      << "Return mismatch with possibly appended outer dimension (" << shape[0]
      << ") of Tensor: \"" << name
      << "\". should match numGroups: " << numGroups << "";

  Shape result{shape.begin() + 1, shape.end()};
  return result;
}

/**
 * Get the shape on host. Used internally in `popart::VariableSettings`, when
 * the `popart::VariableSettings::domain_` is a `popart::ReplicaGrouping`.
 */
static Shape shapeOnHost(const ReplicaGrouping &grouping, const Shape &shape) {
  auto numGroups = grouping.getNumGroups();

  if (numGroups == 1) {
    return shape;
  }

  Shape result = {numGroups};
  result.insert(result.end(), shape.begin(), shape.end());
  return result;
}

/**
 * Get the groups. Used internally in `popart::VariableSettings`, when the
 * `popart::VariableSettings::domain_` is a `popart::ReplicaGrouping`.
 */
static std::vector<std::vector<std::int64_t>>
groups(const ReplicaGrouping &grouping) {
  std::vector<std::vector<std::int64_t>> result;
  result.reserve(grouping.getNumGroups());

  for (std::size_t i = 0; i < grouping.getNumGroups(); i++) {
    auto group = grouping.getReplicasAt(i);
    result.emplace_back(group.begin(), group.end());
  }

  return result;
}

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
    : useCommGroup(true), commGroupType(CommGroupType::All), stride(),
      groupSize(0), retrievalMode(VariableRetrievalMode::OnePerGroup) {}

/// Defaults VariableRetrievalMode to OnePerGroup
VariableSettings::VariableSettings(CommGroup sharedVariableDomain_)
    : useCommGroup(true), commGroupType(sharedVariableDomain_.type), stride(),
      groupSize(sharedVariableDomain_.replicaGroupSize),
      retrievalMode(VariableRetrievalMode::OnePerGroup) {
  verify();
}

/// Defaults CommGroup to [All, 0]
VariableSettings::VariableSettings(VariableRetrievalMode retrievalMode_)
    : useCommGroup(true), commGroupType(CommGroupType::All), stride(),
      groupSize(0), retrievalMode(retrievalMode_) {
  verify();
}

/// Entirely custom VariableSettings
VariableSettings::VariableSettings(CommGroup sharedVariableDomain_,
                                   VariableRetrievalMode retrievalMode_)
    : useCommGroup(true), commGroupType(sharedVariableDomain_.type), stride(),
      groupSize(sharedVariableDomain_.replicaGroupSize),
      retrievalMode(retrievalMode_) {
  verify();
}

VariableSettings::VariableSettings(unsigned stride, unsigned groupSize)
    : useCommGroup(false), commGroupType(), stride(stride),
      groupSize(groupSize), retrievalMode(VariableRetrievalMode::OnePerGroup) {
  verify();
}

VariableSettings::VariableSettings(unsigned stride,
                                   unsigned groupSize,
                                   VariableRetrievalMode retrievalMode)
    : useCommGroup(false), commGroupType(), stride(stride),
      groupSize(groupSize), retrievalMode(retrievalMode) {
  verify();
}

const CommGroup VariableSettings::getSharedVariableDomain() const {
  POPART_CHECK(useCommGroup) << "The variable settings were not initialised "
                                "using a `popart::CommGroup` instance.";
  return CommGroup(commGroupType, groupSize);
}

ReplicaGrouping
VariableSettings::getReplicaGrouping(unsigned numReplicas) const {
  if (useCommGroup) {
    return getSharedVariableDomain().toReplicaGrouping(numReplicas);
  }
  return {numReplicas, stride, groupSize};
}

unsigned
VariableSettings::numReplicasReturningVariable(unsigned replicaCount) const {
  if (!useCommGroup) {
    return ::popart::numReplicasReturningVariable(
        getReplicaGrouping(replicaCount), retrievalMode);
  }

  const auto &sharedVariableDomain = getSharedVariableDomain();

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
  if (!useCommGroup) {
    return ::popart::getGroupCount(getReplicaGrouping(replicaCount));
  }

  const auto &sharedVariableDomain = getSharedVariableDomain();

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
  if (!useCommGroup) {
    return ::popart::getStride(getReplicaGrouping(replicaCount));
  }

  const auto &sharedVariableDomain = getSharedVariableDomain();

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
  if (!useCommGroup) {
    return ::popart::getRealGroupSize(getReplicaGrouping(replicaCount));
  }

  const auto &sharedVariableDomain = getSharedVariableDomain();

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
  if (!useCommGroup) {
    // `numReplicas` should be big enough to so that the `group` exists in the
    // created `popart::ReplicaGrouping`.
    const unsigned numReplicas = groupSize * stride * (group + 1);
    return ::popart::getGroupRepresentative(getReplicaGrouping(numReplicas),
                                            group);
  }

  const auto &sharedVariableDomain = getSharedVariableDomain();

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

  if (useCommGroup) {
    const auto &sharedVariableDomain = getSharedVariableDomain();

    auto type = static_cast<int64_t>(sharedVariableDomain.type);

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
  } else {
    POPART_CHECK_NE(stride, 0)
        << "The stride in a `popart::VariableSettings` must be a positive "
           "integer.";
    POPART_CHECK_NE(groupSize, 0)
        << "The group size in a `popart::VariableSettings` must be a positive "
           "integer.";

    if (groupSize == 1 && stride != 1) {
      stride = 1;
    }
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
  if (!useCommGroup) {
    return ::popart::shapeOnReplica(
        getReplicaGrouping(replicaCount), full_shape, name);
  }

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
  if (!useCommGroup) {
    return ::popart::shapeOnHost(getReplicaGrouping(replicaCount),
                                 replicaShape);
  }

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
  if (!useCommGroup) {
    return ::popart::groups(getReplicaGrouping(replicaCount));
  }

  const auto &sharedVariableDomain = getSharedVariableDomain();

  std::vector<std::vector<std::int64_t>> groups;
  std::vector<std::int64_t> group;
  if (sharedVariableDomain.type == CommGroupType::All) {
    group.resize(replicaCount);
    std::iota(group.begin(), group.end(), 0);
    groups.push_back(group);
    return groups;
  }
  if (sharedVariableDomain.type == CommGroupType::None) {
    static constexpr unsigned group_size = 1;
    groups.reserve(replicaCount);
    for (auto i = 0; i < replicaCount; i++) {
      groups.push_back(std::vector<std::int64_t>(group_size, i));
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

bool VariableSettings::operator==(const VariableSettings &other) const {
  if (useCommGroup != other.useCommGroup) {
    return false;
  }
  if (useCommGroup) {
    return std::tie(commGroupType, groupSize, retrievalMode) ==
           std::tie(other.commGroupType, other.groupSize, other.retrievalMode);
  }
  return std::tie(stride, groupSize, retrievalMode) ==
         std::tie(other.stride, other.groupSize, other.retrievalMode);
}

bool VariableSettings::operator!=(const VariableSettings &other) const {
  return !this->operator==(other);
}

std::ostream &operator<<(std::ostream &os, const VariableSettings &vs) {
  return os << "VariableSettings: [" << vs.getSharedVariableDomain() << ", "
            << vs.getRetrievalMode() << "]";
}
} // namespace popart
