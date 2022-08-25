// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "variablesettingsdomain.hpp"

#include "popart/basicoptionals.hpp"
#include "popart/commgroup.hpp"
#include "popart/replicagrouping.hpp"

namespace popart {

VariableSettingsDomain::VariableSettingsDomain(const ReplicaGrouping &grouping)
    : grouping_(grouping) {}

VariableSettingsDomain::VariableSettingsDomain(unsigned numReplicas,
                                               unsigned stride,
                                               unsigned groupSize)
    : grouping_({numReplicas, stride, groupSize}) {}

VariableSettingsDomain::VariableSettingsDomain(const CommGroup &commGroup)
    : commGroup_(commGroup) {}

VariableSettingsDomain::VariableSettingsDomain(CommGroupType type,
                                               unsigned groupSize)

    : commGroup_({type, groupSize}) {}

bool VariableSettingsDomain::operator==(
    const VariableSettingsDomain &other) const {
  if (commGroup_.has_value() != other.commGroup_.has_value()) {
    return false;
  }

  if (commGroup_.has_value()) {
    return commGroup_.value() == other.commGroup_.value();
  }

  return grouping_.value() == other.grouping_.value();
}

bool VariableSettingsDomain::operator!=(
    const VariableSettingsDomain &other) const {
  return !operator==(other);
}

} // namespace popart
