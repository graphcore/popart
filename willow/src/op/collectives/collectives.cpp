// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

CollectivesBaseOp::CollectivesBaseOp(const OperatorIdentifier &_opid,
                                     const Op::Settings &settings_)
    : Op(_opid, settings_) {}

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

} // namespace popart
