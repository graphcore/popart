// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/commgroup.hpp>
#include <popart/error.hpp>
#include <popart/variablesettings.hpp>
namespace popart {

std::ostream &operator<<(std::ostream &os, const VariableRetrievalMode &vrm) {
  switch (vrm) {
  case VariableRetrievalMode::OnePerGroup:
    os << "OnePerGroup";
    break;
  case VariableRetrievalMode::AllReduceReplicas:
    os << "AllReduceReplicas";
    break;
  case VariableRetrievalMode::AllReplicas:
    os << "AllReplicas";
    break;
  default:
    os << "Undefined VariableRetrievalMode: " << static_cast<int>(vrm);
    break;
  }
  return os;
}

/// "Default" constructor, defaults CommGroup to [All, 0] and retrievalMode to
/// OnePerGroup
VariableSettings::VariableSettings()
    : sharedVariableDomain(CommGroup(CommGroupType::All, 0)),
      retrievalMode(VariableRetrievalMode::OnePerGroup) {
  verify();
}

/// Defaults VariableRetrievalMode to OnePerGroup
VariableSettings::VariableSettings(CommGroup sharedVariableDomain_)
    : sharedVariableDomain(sharedVariableDomain_),
      retrievalMode(VariableRetrievalMode::OnePerGroup) {
  verify();
}

/// Defaults CommGroup to [All, 0]
VariableSettings::VariableSettings(VariableRetrievalMode retrievalMode_)
    : sharedVariableDomain(CommGroup(CommGroupType::All, 0)),
      retrievalMode(retrievalMode_) {
  verify();
}

/// Entirely custom VariableSettings
VariableSettings::VariableSettings(CommGroup sharedVariableDomain_,
                                   VariableRetrievalMode retrievalMode_)
    : sharedVariableDomain(sharedVariableDomain_),
      retrievalMode(retrievalMode_) {
  verify();
}

unsigned VariableSettings::numReplicasReturningVariable(unsigned replicaCount) {

  // If instruction is to return from all,
  // replicas are ungrouped, or if group size
  // is one, we know to return for all replicas

  if (retrievalMode == VariableRetrievalMode::AllReplicas ||
      sharedVariableDomain.type == CommGroupType::None ||
      sharedVariableDomain.replicaGroupSize == 1)
    return replicaCount;

  // Logical Point: Beyond this we know retrievalMode is
  // OnePerGroup or AllReduceReplicas, meaning we will return
  //   at most one per domain/group.

  // Make sure the dimension of the commGroup is a valid number.
  if (sharedVariableDomain.type != CommGroupType::All &&
      sharedVariableDomain.replicaGroupSize < 1) {
    logging::err("Attempting to use 0 as a CommGroupSize, "
                 "groups of type {} must have legal size.\n",
                 sharedVariableDomain.type);
  }

  // differentiate between options returning one and group-size.
  // Also
  switch (sharedVariableDomain.type) {
  case CommGroupType::All:
    return 1;
  case CommGroupType::Consecutive:
  case CommGroupType::Orthogonal:
    return replicaCount / sharedVariableDomain.replicaGroupSize;
  case CommGroupType::None:
    logging::err(
        "Unreachable point, logic should have circumvented this case.\n");
    break;
  case CommGroupType::N:
  default:
    logging::err("Bad CommGroupType {} in VariableSetting.\n",
                 sharedVariableDomain.type);
    break;
  }

  logging::err("Unresolved switch-case, logic should have circumvented it.\n");
  return -1;
}

unsigned VariableSettings::getGroupRepresentative(unsigned group) {
  switch (sharedVariableDomain.type) {
  case CommGroupType::All:
    return 0;

  case CommGroupType::Consecutive:
    return sharedVariableDomain.replicaGroupSize * group;

  case CommGroupType::Orthogonal:
  case CommGroupType::None:
    return group;

  default:
    logging::err("Bad CommGroupType {} in VariableSetting.\n",
                 sharedVariableDomain.type);
    return -1;
  }
}

void VariableSettings::verify() {
  logging::debug("Verifying VariableSettings: (CommGroup=(.type={}, .size={}), "
                 "VariableRetrievalMode={})",
                 sharedVariableDomain.type,
                 sharedVariableDomain.replicaGroupSize,
                 retrievalMode);

  int throw_error = 0;
  auto type       = static_cast<int64_t>(sharedVariableDomain.type);

  if (type < 0 || type >= static_cast<int64_t>(CommGroupType::N)) {
    throw_error++;
    logging::err("Bad Commgroup: ", sharedVariableDomain.type);
  }

  if (sharedVariableDomain.type != CommGroupType::All &&
      sharedVariableDomain.type != CommGroupType::None &&
      sharedVariableDomain.replicaGroupSize < 1) {
    throw_error++;
    logging::err("Bad ReplicaGroupSize ({}) for domain type ({})!",
                 sharedVariableDomain.replicaGroupSize,
                 sharedVariableDomain.replicaGroupSize);
  }

  if (retrievalMode != VariableRetrievalMode::OnePerGroup &&
      retrievalMode != VariableRetrievalMode::AllReduceReplicas &&
      retrievalMode != VariableRetrievalMode::AllReplicas) {
    throw_error++;
    logging::err("Bad VariableRetrievalMode: {}", retrievalMode);
  }

  if (throw_error) {
    throw internal_error("VariableSettings had {} errors!", throw_error);
  }
}

std::ostream &operator<<(std::ostream &os, VariableSettings &vs) {
  return os << "VariableSettings: [" << vs.getSharedVariableDomain() << ", "
            << vs.getRetrievalMode() << "]";
}
} // namespace popart