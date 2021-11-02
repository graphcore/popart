// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/commgroup.hpp>
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
    : sharedVariableDomain(CommGroup()),
      retrievalMode(VariableRetrievalMode::OnePerGroup) {}

/// Defaults VariableRetrievalMode to OnePerGroup
VariableSettings::VariableSettings(CommGroup sharedVariableDomain_)
    : sharedVariableDomain(sharedVariableDomain_),
      retrievalMode(VariableRetrievalMode::OnePerGroup) {}

// Defaults CommGroup to [All, 0]
VariableSettings::VariableSettings(VariableRetrievalMode retrievalMode_)
    : sharedVariableDomain(CommGroup()), retrievalMode(retrievalMode_) {}

// Entirely custom VariableSettings
VariableSettings::VariableSettings(CommGroup sharedVariableDomain_,
                                   VariableRetrievalMode retrievalMode_)
    : sharedVariableDomain(sharedVariableDomain_),
      retrievalMode(retrievalMode_) {}

std::ostream &operator<<(std::ostream &os, VariableSettings &vs) {
  return os << "VariableSettings: [" << vs.getSharedVariableDomain() << ", "
            << vs.getRetrievalMode() << "]";
}
} // namespace popart