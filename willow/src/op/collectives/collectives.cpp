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

CommGroup::CommGroup() = default;

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
    throw error("Unsupported CommGroupType {}",
                std::underlying_type_t<CommGroupType>(commType));
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const CommGroup &group) {
  os << "CommGroup(type=" << group.type
     << ", replicaGroupSize=" << group.replicaGroupSize << ")";
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

::gcl::CommGroup toGCLCommGroup(const ::popart::CommGroup &group) {
  ::gcl::CommGroupType type;
  switch (group.type) {
  case ::popart::CommGroupType::All:
    type = ::gcl::CommGroupType::ALL;
    break;
  case ::popart::CommGroupType::Consecutive:
    type = ::gcl::CommGroupType::CONSECUTIVE;
    break;
  case ::popart::CommGroupType::Orthogonal:
    type = ::gcl::CommGroupType::ORTHOGONAL;
    break;
  default:
    throw error("Cannot convert unknown CommGroup type");
  }
  return {type, group.replicaGroupSize};
}

} // namespace popart
