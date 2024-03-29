// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef POPART_WILLOW_INCLUDE_POPART_VARIABLESETTINGS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_VARIABLESETTINGS_HPP_

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <vector>
#include <popart/commgroup.hpp>
#include <popart/names.hpp>
#include <popart/replicagrouping.hpp>
#include <popart/util/expressionchecking.hpp>

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
 * A class to dictate behaviour of variables and reductions of such across
 * multiple graphs.
 */
class VariableSettings {
private:
  /**
   * Depending on which constructor was used, the `popart::VariableSettings`
   * will either store information for constructing a `popart::CommGroup` or a
   * `popart::ReplicaGrouping`. This flag is used to indicate this.
   */
  bool useCommGroup;

  /**
   * When `popart::VariableSettings::useCommGroup` is false, the value in this
   * member has no meaning.
   */
  CommGroupType commGroupType;

  /**
   * When `popart::VariableSettings::useCommGroup` is true, the value in this
   * member has no meaning.
   */
  unsigned stride;

  /**
   * The group size used when constructing a `popart::CommGroup` or a
   * `popart::ReplicaGrouping` in this class' methods.
   */
  unsigned groupSize;

  /**
   * Dictates how Variable retrieval is conducted.
   */
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
  const CommGroup getSharedVariableDomain() const;

  /**
   * \param numReplicas The number of replicas in the IR this is used in.
   * \return the ReplicaGrouping domain of this VariableSettings.
   */
  ReplicaGrouping getReplicaGrouping(unsigned numReplicas) const;

  /**
   * \return whether the VariableSettings were initialised using a CommGroup or
   *         a stride.
   */
  bool isUsingCommGroup() const;

  /**
   * \return the CommGroupType. The value of this is invalid if
   *         VariableSettings::isUsingCommGroup returns false.
   */
  CommGroupType getCommGroupType() const;

  /**
   * \return the stride. The value of this is invalid if
   *         VariableSettings::isUsingCommGroup returns true.
   */
  unsigned getStride() const;

  /**
   * \return the replica group size.
   */
  unsigned getGroupSize() const;

  /**
   * \return the VariableRetrievalMode retrievalMode of this VariableSettings.
   */
  VariableRetrievalMode getRetrievalMode() const { return retrievalMode; }

  /**
   *  "Default" constructor, defaults CommGroup to [All, 0] and retrievalMode to
   * OnePerGroup.
   */
  VariableSettings();

  /**
   *  Defaults VariableRetrievalMode to OnePerGroup.
   */
  VariableSettings(CommGroup sharedVariableDomain_);

  /**
   *  Defaults CommGroup to [All, 0].
   */
  VariableSettings(VariableRetrievalMode retrievalMode_);

  /**
   *  Entirely custom VariableSettings.
   */
  VariableSettings(CommGroup sharedVariableDomain_,
                   VariableRetrievalMode retrievalMode_);

  VariableSettings(unsigned stride, unsigned groupSize);

  VariableSettings(unsigned stride,
                   unsigned groupSize,
                   VariableRetrievalMode retrievalMode);

  /**
   * Calculate the number of replicas that will
   * return this variable
   * \param  replicaCount Number of global replicas.
   * \return              Number of variables returned.
   */
  unsigned numReplicasReturningVariable(unsigned replicaCount) const;

  /**
   * \param replicaCount The replicationFactor of the graph.
   * \return             The number of groups given the replicaFactor and the
   *                     VariableSettings.
   */
  unsigned getGroupCount(unsigned replicaCount) const;

  /**
   * \param replicaCount The replicationFactor of the graph.
   * \return             The stride between each member of a group.
   */
  unsigned getStride(unsigned replicaCount) const;

  /**
   * Because CommGroup's don't have a defined group-size if
   * the type is All or None, this function will return a
   * group-size that is _always_ accurate, based on replicas.
   * \param replicaCount The replication factor
   * \return             The actual number of replicas in a group
   *
   */
  unsigned getRealGroupSize(unsigned replicaCount) const;

  /**
   * Get the default \a first member of a group
   * \param  group The group to return the representative for.
   * \return       The representative replica of this group.
   */
  unsigned getGroupRepresentative(unsigned group) const;

  /**
   * The shape Onnx reads holds an extra outer dimension in certain cases,
   * where the outer dimension represents the number of returning replica
   * variables. This function takes an Onnx full-shape and removes the outer
   * dimension safely (ie. checks if the outer dimension matches an expected
   * outer dimension). A quick-function to avoid duplicate code.
   * \param  full_shape   The shape as presented by Onnx.
   * \param  replicaCount The local replication factor, used to calculate
   *                      the return factor.
   * \param  name         The TensorId of the function, used to give good
   *                      error feedback.
   * \return              The shape of the data on the replica.
   */
  Shape shapeOnReplica(Shape full_shape,
                       unsigned replicaCount,
                       const TensorId name) const;

  /**
   * Takes the shape of a tensor on a replica and returns it's full ONNX shape.
   *
   * This is the inverse operation to shapeOnReplica
   *
   * \param  replica_shape The shape of the data on a replica.
   * \param  replicaCount  The local replication factor, used to calculate
   *                       the return factor.
   * \return               The shape as presented by Onnx.
   */
  Shape shapeOnHost(Shape replica_shape, unsigned replicaCount) const;

  /**
   * This function returns a set of vectors where each vector contains all
   * the replicaId's of the replicas with a sharedVariableDomain given the
   * variableSettings and the replicaCount.
   *
   * \param  replicaCount The local replication factor
   * \return              A set of sets, such that set.at(a).set(b) is member
   *                      nr. b of group a, and set.size() is the number og
   *                      groups and set.at(A).size() is the size of the group.
   */
  std::vector<std::vector<std::int64_t>> groups(unsigned replicaCount) const;

  /**
   * Compare two variable-settings
   * \param other VariableSettings to compare these settings to.
   * \return      True if all internal elements are the same
   */
  bool operator==(const VariableSettings &other) const;

  /**
   * Compare two variable-settings
   * \param other VariableSettings to compare these settings to.
   * \return      False if all internal elements are the same
   *
   */
  bool operator!=(const VariableSettings &other) const;
};

/**
 * \param os Stream to append VariableRetrievalMode vrm to.
 * \param vrm VariableRetrievalMode to add to the stream.
 * \return Input stream with vrm appended to the end of it
 */
std::ostream &operator<<(std::ostream &os, const VariableRetrievalMode &vrm);

/**
 * \param os Stream to append VariableSettings vrm to.
 * \param vs VariableSettings to add to the stream.
 * \return Input stream with vs appended to the end of it
 */
std::ostream &operator<<(std::ostream &os, const VariableSettings &vs);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_VARIABLESETTINGS_HPP_
