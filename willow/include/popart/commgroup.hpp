// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_COMMGROUP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_COMMGROUP_HPP_

#include <iostream>

namespace popart {

class ReplicaGrouping;

/// PopART equivalent of GCL CommGroupType. Each of these enumeration constants
/// have a corresponding GCL CommGroupType value.
enum class CommGroupType {
  /** All replicas viewed as one group, replica group size is ignored. */
  All = 0,

  /** Groups are consecutive in replica.
   * If there are N replicas denoted {0, ... N-1} and group size is k,
   * then there are N/k groups of size k:
   *   {0, 1, ... k-1}, {k, ... 2k-1} ... {N-k-1, ... N-1}
   */
  Consecutive,

  /** Groups are sliced orthogonal to the replica ordering.
   * If there are N replicas denoted {0, ... N-1} and group size is k,
   * then there are m = N/k groups of size k:
   *   {0, m, 2m, ...}, {1, m+1, 2m+1, ...} ... {m-1, 2m-1, ... N-1}
   */
  Orthogonal,

  /** Each replica is in it's own group, replica group size is ignored. */
  None,

  /**
   * Number of values
   */
  N
};

/** Class to specify sub-groups of replicas.
 *
 * Examples of derived sub-groups:
 * - IPU-link domain sub-rack:
 *
 *   .. code-block:: python
 *     type == Consecutive && replicaGroupSize == 64/replica-size/N
 *
 *   where N is power of two and replicaGroupSize > 1.
 *
 * - Complete IPU-link domain / full rack:
 *
 *   .. code-block:: python
 *     type == Consecutive && replicaGroupSize == 64/replica-size
 *
 * - Using GW-links only:
 *
 *   .. code-block:: python
 *     type == Orthogonal && replicaGroupSize == numberOfIpuLinkDomains
 *
 */
class CommGroup {
public:
  /**
   * Default CommGroup constructor.
   *
   * Sets type to CommGroupType::All and replicaGroupSize to 0.
   */
  CommGroup();

  /**
   * Construct CommGroup
   *
   * \param groupType replica group type
   * \param groupSize replica group size
   */
  CommGroup(CommGroupType type, unsigned groupSize)
      : type(type), replicaGroupSize(groupSize) {}

  /**
   * Construct CommGroup from a ReplicaGrouping.
   *
   * \param grouping The replica grouping.
   */
  explicit CommGroup(const ReplicaGrouping &grouping);

  /**
   * Convert this CommGroup to a ReplicaGrouping.
   *
   * \param numReplicas The number of replicas to pass to create the replica
   *   grouping with.
   * \return The replica grouping.
   */
  ReplicaGrouping toReplicaGrouping(unsigned numReplicas) const;

  bool operator==(const CommGroup &other) const;
  bool operator!=(const CommGroup &other) const;

  /**
   * Replica group type.
   */
  CommGroupType type = CommGroupType::All;

  /**
   * Replica group size.
   */
  unsigned replicaGroupSize = 0;
};

std::ostream &operator<<(std::ostream &os, CommGroupType commType);
std::ostream &operator<<(std::ostream &os, const CommGroup &group);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_COMMGROUP_HPP_
