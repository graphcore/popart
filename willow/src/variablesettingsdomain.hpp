// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_VARIABLESETTINGSDOMAIN_HPP_
#define POPART_WILLOW_SRC_VARIABLESETTINGSDOMAIN_HPP_

#include "popart/commgroup.hpp"
#include "popart/replicagrouping.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {

/**
 * A union, which can be initialised with one of `popart::CommGroup` or
 * `popart::ReplicaGrouping`.
 */
struct VariableSettingsDomain {
  const nonstd::optional<CommGroup> commGroup_;
  const nonstd::optional<ReplicaGrouping> grouping_;

  VariableSettingsDomain(const ReplicaGrouping &grouping);

  VariableSettingsDomain(unsigned numReplicas,
                         unsigned stride,
                         unsigned groupSize);

  VariableSettingsDomain(const CommGroup &commGroup);

  VariableSettingsDomain(CommGroupType type, unsigned groupSize);

  bool operator==(const VariableSettingsDomain &other) const;

  bool operator!=(const VariableSettingsDomain &other) const;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_VARIABLESETTINGSDOMAIN_HPP_
