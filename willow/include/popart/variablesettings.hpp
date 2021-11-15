// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef GUARD_VARIABLE_SETTINGS_HPP
#define GUARD_VARIABLE_SETTINGS_HPP

#include <string>

#include <popart/commgroup.hpp>

namespace popart {

/**
 * Enum type that describes how to retrieve variables from
 * the replicas. Each replica is in a group defined by
 * the \c VariableSettings::sharedVariableDomain. Replicas
 * within a group have variables initialized with the same
 * values.
 */
enum class VariableRetrievalMode {
  /**
   * Returns one variable per group (defined by the
   * \c VariableSettings::sharedVariableDomain \c CommGroup),
   * automatically returns the \a first replica of each group,
   * where \a first means the one with the lowest replica ID.
   */
  OnePerGroup = 0,

  /**
   * As OnePerGroup, but performs an AllReduce among the
   * replicas in the same group according to
   * \c VariableSettings::sharedVariableDomain
   * !!! CURRENTLY UNSUPPORTED
   */
  AllReduceReplicas,

  /**
   * Returns all replica Weights
   */
  AllReplicas
};

/**
 * \param os Stream to append VariableRetrievalMode vrm to.
 * \param vrm VariableRetrievalMode to add to the stream.
 * \return Input stream with vrm appended to the end of it
 */
std::ostream &operator<<(std::ostream &os, VariableRetrievalMode &vrm);

class VariableSettings {
private:
  CommGroup sharedVariableDomain      = CommGroup(CommGroupType::All, 0);
  VariableRetrievalMode retrievalMode = VariableRetrievalMode::OnePerGroup;

public:
  /**
   * Runs test to see if the VariableSettings are invalid, and throws an error
   * if so.
   */
  void verify();

  /**
   * \return the CommGroup sharedVariableDomain of this VariableSettings.
   */
  const CommGroup getSharedVariableDomain() const {
    return sharedVariableDomain;
  }
  /**
   * \return the VariableRetrievalMode retrievalMode of this VariableSettings.
   */
  VariableRetrievalMode getRetrievalMode() const { return retrievalMode; }

  /// "Default" constructor, defaults CommGroup to [All, 0] and retrievalMode to
  /// OnePerGroup
  VariableSettings();

  /// Defaults VariableRetrievalMode to OnePerGroup
  VariableSettings(CommGroup sharedVariableDomain_);

  // Defaults CommGroup to [All, 0]
  VariableSettings(VariableRetrievalMode retrievalMode_);

  // Entirely custom VariableSettings
  VariableSettings(CommGroup sharedVariableDomain_,
                   VariableRetrievalMode retrievalMode_);

  /**
   * Calculate the number of replicas that will
   * return this variable
   * \param replicaCount Number of global replicas
   * \return Number of variables returned
   */
  unsigned numReplicasReturningVariable(unsigned replicaCount);

  /**
   * Get the default \a first member of a group
   * \param group The group to return the representative for.
   * \return the representative replica of this group
   */
  unsigned getGroupRepresentative(unsigned group);
};
} // namespace popart

#endif