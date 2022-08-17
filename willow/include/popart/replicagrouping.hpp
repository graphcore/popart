// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPART_WILLOW_INCLUDE_POPART_REPLICAGROUPING_HPP_
#define POPART_WILLOW_INCLUDE_POPART_REPLICAGROUPING_HPP_

#include <string>
#include <vector>

namespace popart {

/**
 * A class to represent groups of replicas using a group size and a stride.
 */
class ReplicaGrouping {
private:
  unsigned numReplicas_;
  unsigned stride_;
  unsigned groupSize_;

  void checkReplicaIsValid(unsigned replica) const;
  void checkGroupIsValid(unsigned group) const;
  void checkIndexInGroupIsValid(unsigned index) const;

public:
  ReplicaGrouping(unsigned numReplicas, unsigned stride, unsigned groupSize);

  ReplicaGrouping(unsigned numReplicas);

  /**
   * Get the number of replicas.
   *
   * \return The number of replicas.
   */
  unsigned getNumReplicas() const;

  /**
   * Get the stride.
   *
   * \return The stride.
   */
  unsigned getStride() const;

  /**
   * Get the group size.
   *
   * \return The group size.
   */
  unsigned getGroupSize() const;

  /**
   * Get the number of groups.
   *
   * \return The number of groups.
   */
  unsigned getNumGroups() const;

  /**
   * Get the group to which the given replica belongs.
   *
   * \param replica The replica for which a group will be returned.
   * \throws popart::error if the replica index is not a part of this grouping.
   * \return The group to which the given replica belongs.
   */
  unsigned getGroupAt(unsigned replica) const;

  /**
   * Get the index of the given replica within its group.
   *
   * \param replica The replica for which an index will be returned.
   * \throws popart::error if the replica index is not a part of this grouping.
   * \return The index of the given replica within its group.
   */
  unsigned getIndexInGroupAt(unsigned replica) const;

  /**
   * Get the replica in the given group at the given index within that group.
   *
   * \param group The group to which the replica belongs.
   * \param index The index within that group.
   * \throws popart::error if the group index is not a part of this grouping.
   * \throws popart::error if the index is outside the limit of the group.
   * \return The replica which belongs to the given group at the given index.
   */
  unsigned getReplicaAt(unsigned group, unsigned index = 0) const;

  /**
   * Get a vector of replicas which belong to the given group.
   *
   * \param group The group for which a list of replicas will be returned.
   * \throws popart::error if the group index is not a part of this grouping.
   * \return The replicas which belong to the given group.
   */
  std::vector<unsigned> getReplicasAt(unsigned group) const;

  /**
   * Create the transpose of this replica grouping.
   *
   * The transpose of a replica grouping is such that the group index of a
   * replica in the transpose is equal to the index of that replica within its
   * original group.
   *
   * A good way of visualising what the transpose of a replica grouping is the
   * following. Consider the following representation of a replica grouping with
   * 8 replicas, a stride of 2, and a group size of 4. The first dimension
   * represents the group index. The second dimension contains the replica
   * indices in that group.
   *
   * ```
   * [[0, 2, 4, 6],
   *  [1, 3, 5, 7]]
   * ```
   *
   * Transposing the matrix above results in the representation of the transpose
   * of the replica grouping it represents, which now has a stride of 1, and a
   * group size of 2.
   *
   * ```
   * [[0, 1],
   *  [2, 3],
   *  [4, 5],
   *  [6, 7]]
   * ```
   *
   * There are two important properties of the transpose:
   * - Transposing is symmetric. Transposing a replica grouping twice will
   *   result in the same replica grouping.
   * - All replicas that are in the same group, cannot be in the same group in
   *   the transposed replica grouping.
   *
   * Note that the transpose of some replica groupings cannot be represented by
   * a replica grouping. For example, trying to transpose a grouping with 12
   * replicas, a stride of 3 and a group size of 2 will result in an error.
   *
   * \throws popart::error if the transpose cannot be represented by a replica
   *   grouping.
   # \return The transpose of this replica grouping.
   */
  ReplicaGrouping getTranspose() const;

  /**
   * Get a string representation of the replica grouping.
   *
   * \return The string representation of the replica grouping.
   */
  std::string str() const;

  bool operator==(const ReplicaGrouping &other) const;

  bool operator!=(const ReplicaGrouping &other) const;
};

std::ostream &operator<<(std::ostream &os, const ReplicaGrouping &grouping);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_REPLICAGROUPING_HPP_
