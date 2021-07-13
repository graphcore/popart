// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COLLECTIVES_HPP
#define GUARD_NEURALNET_COLLECTIVES_HPP

#include <popart/op.hpp>

namespace gcl {
class CommGroup;
} // namespace gcl

namespace popart {

enum class CollectiveOperator {
  Add = 0,
  Mean,
  Mul,
  Min,
  Max,
  LogicalAnd,
  LogicalOr,
  SquareAdd,
  Local,
  N
};

std::ostream &operator<<(std::ostream &os, const CollectiveOperator &op);

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
  N
};

/** Struct to specify sub-groups of replicas.
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
struct CommGroup {
  CommGroup();

  /**
   * Construct CommGroup
   *
   * \param groupType replica group type
   * \param groupSize replica group size
   */
  CommGroup(CommGroupType type, unsigned groupSize)
      : type(type), replicaGroupSize(groupSize) {}

  /** Replica group type */
  CommGroupType type = CommGroupType::All;

  /** Replica group size */
  unsigned replicaGroupSize = 0;
};

std::ostream &operator<<(std::ostream &os, CommGroupType commType);
std::ostream &operator<<(std::ostream &os, const CommGroup &group);

class CollectivesBaseOp : public Op {
public:
  CollectivesBaseOp(const OperatorIdentifier &_opid,
                    CommGroup group,
                    const Op::Settings &settings_);

  // Input to gather/reduce/scatter
  static InIndex getInIndex() { return 0; }

  // Tensor to backtrack collective ops that have to coordinate with each other
  static InIndex getCollectiveLinkedIndex() { return 1; }

  // Gathered/reduced/scattered output
  static OutIndex getOutIndex() { return 0; }
  CommGroup getGCLCommGroup() const { return group; }

  void appendOutlineAttributes(OpSerialiserBase &os) const override;

private:
  CommGroup group;
};

/**
 * Extracts \ref CommGroup from op's attributes. If the attribute isn't set,
 * then the function returns a default constructed \ref CommGroup.
 *
 * \param attrs Op's attributes.
 * \return \ref CommGroup that is extracted from attributes.
 */
CommGroup extractCommGroupFromAttrs(const Attributes &attrs);

/**
 * Extracts \ref CommGroup from vector of two integers. If the vector is empty,
 * then the function returns a default constructed \ref CommGroup.
 *
 * \param vec Vector of two integers corresponding to the \ref CommGroupType and
 * replicaGroupSize.
 * \return \ref CommGroup that is extracted from the input vector.
 */
CommGroup extractCommGroupFromVector(const std::vector<int64_t> &vec);

/**
 * Converts give \ref CommGroup to GCL's CommGroup type.

 * \param input PopART \ref CommGroup.
 * \return GCL CommGroup.
 */
::gcl::CommGroup toGCLCommGroup(const ::popart::CommGroup &input);

} // namespace popart

#endif
