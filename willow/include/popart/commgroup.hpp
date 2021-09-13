// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COMMGROUPS_HPP
#define GUARD_NEURALNET_COMMGROUPS_HPP

#include <iostream>

namespace popart {

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

  /**
   * Empty group, used as a complement group to \a CommGroupType::All
   */
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
 *   \code
 *     type == Consecutive && replicaGroupSize == 64/replica-size/N
 *   \endcode
 *   where N is power of two and replicaGroupSize > 1.
 * - Complete IPU-link domain / full rack:
 *   \code
 *     type == Consecutive && replicaGroupSize == 64/replica-size
 *   \endcode
 * - Using GW-links only:
 *   \code
 *     type == Orthogonal && replicaGroupSize == 64/replica-size
 *   \endcode
 */
class CommGroup {
public:
  CommGroup();

  /**
   * Construct CommGroup
   *
   * \param groupType replica group type
   * \param groupSize replica group size
   */
  CommGroup(CommGroupType type, unsigned groupSize)
      : type(type), replicaGroupSize(groupSize) {}

  bool operator==(const CommGroup &other) const;
  bool operator!=(const CommGroup &other) const;

  /** Replica group type */
  CommGroupType type = CommGroupType::All;

  /** Replica group size */
  unsigned replicaGroupSize = 0;
};

std::ostream &operator<<(std::ostream &os, CommGroupType commType);
std::ostream &operator<<(std::ostream &os, const CommGroup &group);

} // namespace popart

#endif
