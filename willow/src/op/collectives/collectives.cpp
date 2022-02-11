// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <gcl/Collectives.hpp>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

CollectivesBaseOp::CollectivesBaseOp(const OperatorIdentifier &_opid,
                                     CommGroup group,
                                     const Op::Settings &settings_)
    : Op(_opid, settings_), group(group) {}

void CollectivesBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute(
      sCollectiveCommGroup,
      std::vector<int64_t>{
          static_cast<std::underlying_type_t<CommGroupType>>(group.type),
          group.replicaGroupSize});
}

std::ostream &operator<<(std::ostream &os, const CollectiveOperator &op) {
  switch (op) {
  case CollectiveOperator::Add:
    os << "Add";
    break;
  case CollectiveOperator::Mean:
    os << "Mean";
    break;
  case CollectiveOperator::Mul:
    os << "Mul";
    break;
  case CollectiveOperator::Min:
    os << "Min";
    break;
  case CollectiveOperator::Max:
    os << "Max";
    break;
  case CollectiveOperator::LogicalAnd:
    os << "LogicalAnd";
    break;
  case CollectiveOperator::LogicalOr:
    os << "LogicalOr";
    break;
  case CollectiveOperator::SquareAdd:
    os << "SquareAdd";
    break;
  case CollectiveOperator::Local:
    os << "Local";
    break;
  default:
    throw error("Unsupported CollectiveOperator {}", static_cast<int>(op));
  }
  return os;
}

CommGroup extractCommGroupFromVector(const std::vector<int64_t> &vec) {
  using IntegerCommType = std::underlying_type_t<CommGroupType>;
  static const std::array<IntegerCommType, 3> knownTypeValues{
      static_cast<IntegerCommType>(CommGroupType::All),
      static_cast<IntegerCommType>(CommGroupType::Consecutive),
      static_cast<IntegerCommType>(CommGroupType::Orthogonal)};
  CommGroupType type = CommGroupType::All;
  unsigned groupSize = 0;
  if (!vec.empty()) {
    if (vec.size() != 2) {
      throw error("Invalid commGroup data for collective op");
    } else {
      int64_t typeArg = vec[0];
      if (!std::any_of(knownTypeValues.cbegin(),
                       knownTypeValues.cend(),
                       [typeArg](IntegerCommType knownType) {
                         return knownType == typeArg;
                       })) {
        throw error("Unknown commGroup type for collective op");
      }
      type = static_cast<CommGroupType>(typeArg);

      if (static_cast<uint64_t>(vec[1]) >
          std::numeric_limits<unsigned>::max()) {
        throw error("Replica group size in commGroup is too large");
      }
      groupSize = vec[1];
    }
  }
  return CommGroup(type, groupSize);
}

CommGroup extractCommGroupFromAttrs(const Attributes &attrs) {
  const std::vector<int64_t> commGroupInfo =
      attrs.getAttribute<Attributes::Ints>(sCollectiveCommGroup, {});
  return extractCommGroupFromVector(commGroupInfo);
}

CommGroup getComplementCommGroup(const Ir &ir, CommGroup group) {
  auto numReplicas = ir.getSessionOptions().getGlobalReplicationFactor();
  switch (group.type) {
  case CommGroupType::Consecutive:
    return CommGroup(CommGroupType::Orthogonal,
                     numReplicas / group.replicaGroupSize);
  case CommGroupType::Orthogonal:
    return CommGroup(CommGroupType::Consecutive,
                     numReplicas / group.replicaGroupSize);
  case CommGroupType::None:
    return CommGroup(CommGroupType::All, 0);
  case CommGroupType::All:
  default:
    return CommGroup(CommGroupType::None, 0);
  }
}

CommGroup getComplementCommGroupWithSuperSet(const Ir &ir,
                                             CommGroup group,
                                             CommGroup superSet) {
  // make the relationship between this function and its sibling very clear.
  if (superSet.type == CommGroupType::All) {
    return getComplementCommGroup(ir, group);
  }

  // Currently the only complement that works if the super-set is not All, is
  // None. Only check replicaGroupSize if replicaGroupSize is readable (that is
  // not the case with CommGroupType::None)
  if ((superSet.type == CommGroupType::None &&
       group.type == CommGroupType::None) ||
      (group.type == superSet.type &&
       group.replicaGroupSize == superSet.replicaGroupSize)) {
    return CommGroup(CommGroupType::None, 0);
  }
  // While there are legitimate logical complements in all cases where the
  // super-set is larger than the group, we still throw because they are not
  // supported in further logic.
  throw internal_error("Could not return a supported CommGroup complement of "
                       "{} within the super-set: {}",
                       group,
                       superSet);
}

} // namespace popart
