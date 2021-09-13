// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COLLECTIVES_HPP
#define GUARD_NEURALNET_COLLECTIVES_HPP

#include <popart/commgroup.hpp>
#include <popart/op.hpp>

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

  void setGCLCommGroup(CommGroup group_) { group = group_; }
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
 * Calculates the complementary group such that the two CommGroups together
 * span all replicas
 */
CommGroup getComplementCommGroup(const Ir &ir, CommGroup group);

} // namespace popart

#endif
