// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef NEURALNET_ANALYSIS_REPLICA_EQUAL_ANALYSIS_RESULTS_HPP
#define NEURALNET_ANALYSIS_REPLICA_EQUAL_ANALYSIS_RESULTS_HPP

#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "popart/graphid.hpp"
#include "popart/names.hpp"
#include "popart/pointercomparators.hpp"
#include "popart/vendored/optional.hpp" // IWYU pragma: keep

namespace popart {
class Op;
class Tensor;

/**
 * This class stores replica-equal values for tensors.
 *
 * Note that a tensor's replica-equal value can change over time due to PopART's
 * Ops having the ability to modify their inputs. Such a modification can change
 * the input tensor's value from one that is replica-equal to one that is not.
 * This class tracks tensor replica-equalness for each possible tensor value
 * in the IR, and has some additional capabilities to help, e.g., fixpoint
 * detection, or the detection of disagreements between call sites.
 *
 * NOTE: This class could be made more general (using templates).
 **/
class ReplicaEqualAnalysisResults {
public:
  // Value of ReplicaEqualAnalysisResults::initTime means "at graph
  // initialisation" time and a value of i for i>0 means "following the
  // execution of Op at schedule index i" onwards. This Op would either be the
  // tensor's producer or an Op modifying the tensor.
  using Time = int;
  // Sparse mapping from time values to replica-equal values. We order this
  // in descending time order so that we can use std::map::upper_bound.
  using ReplicaEqualOverTime =
      std::map<Time, IsReplicaEqual, std::greater<Time>>;
  // A mapping from tensors to replica-equal values over time.
  using Results = std::map<const Tensor *, ReplicaEqualOverTime, PTensorCmp>;
  // A lookup map from Ops to Time values.
  using OpTimeMap = std::map<const Op *, Time, POpCmp>;

  // Type for graph schedules.
  using GraphSchedules = std::map<GraphId, std::vector<Op *>>;
  // Type for disagreements.
  using Disagreements = std::set<const Tensor *>;

  // Constructor.
  ReplicaEqualAnalysisResults();
  virtual ~ReplicaEqualAnalysisResults();

  /**
   * Check if we have a replica-equal value. If Op is set and this Op lives at
   * schedule index i, then this lookup looks for any entry for `tensor` at Time
   * <i. If Op is not set, this lookup looks for entry for `tensor` at
   * ReplicaEqualAnalysisResults::initTime.
   **/
  bool containsAt(const Tensor *tensor, nonstd::optional<const Op *> op) const;

  /**
   * Check if we have a replica-equal value. If the Op lives at schedule
   * index i, then this lookup looks for any entry for `tensor` at Time <i.
   **/
  bool containsBefore(const Tensor *tensor, const Op *op) const;

  /**
   * Set a replica-equal value for a tensor. If the Op is set and this Op lives
   * at schedule index i, an entry is added for `tensor` at time i. If the Op is
   * not set it is added at time initTime.
   *
   * If a value is already set, the incoming value is logically AND'ed with the
   * existing value. If this changes the value (i.e. existing value was 'true'
   * and incoming value is 'false'; resulting value is 'false') a hasChangedFlag
   * is set and if the incoming value differs from the resulting value (i.e.
   * existing value was 'false' and incoming value is 'true'; resulting value is
   * 'false') hasDisagreementFLag flag is set.
   *
   * \param tensor Tensor for which to set the replica-equal value.
   * \param op Op that produced or modified the tensor value (leave unset for
   *    values that exist at initialisation time).
   * \param value The replica-equal value to set.
   * \return true if the value was not already set.
   * \return false if the value was already set.
   */
  bool setValueAt(const Tensor *tensor,
                  nonstd::optional<const Op *> op,
                  const IsReplicaEqual &value);

  /**
   * Get the replica-equal value. If the Op is set and this Op lives at schedule
   * index i, we look for an entry at time i. If the Op is not set we look for
   * an entry at initialisation time ReplicaEqualAnalysisResults::initTime.
   *
   * \param tensor Tensor for which to get the replica-equal value.
   * \param op Op that is to consume the tensor.
   * \return Whether the tensor is replica-equal
   */
  IsReplicaEqual getValueAt(const Tensor *tensor,
                            nonstd::optional<const Op *> op) const;

  /**
   * Get the replica-equal value. If the Op lives at schedule
   * index i, then this lookup looks for any entry for `tensor` at Time <i.
   *
   * \param tensor Tensor for which to get the replica-equal value.
   * \param op Op that is to consume the tensor.
   * \return Whether the tensor is replica-equal
   */
  IsReplicaEqual getValueBefore(const Tensor *tensor, const Op *op) const;

  /**
   * Get the replica-equal value at the latest recorded time index.
   *
   * \param tensor Tensor for which to get the replica-equal value.
   * \return Whether the tensor is replica-equal
   */
  IsReplicaEqual getFinalValue(const Tensor *tensor) const;

  /**
   * Checks if any values have changed since the last call to
   * `clearChanges` call.
   *
   * \return true Since the last call to `clearChanges` a
   *    tensor was set to 'false' whilst previously being 'true'.
   * \return false Since the last call to `clearChanges` a
   *    no tensor was set to 'false' whilst previously being 'true'.
   */
  bool hasChanged() const;

  /**
   * Checks if any values are conflicting since the last call to
   * `clearChanges` call.
   *
   * \return true Since the last call to `clearChanges` a
   *    tensor was set to 'true' whilst already being 'false'.
   * \return false Since the last call to `clearChanges` a
   *    no tensor was set to 'true' whilst already being 'false'.
   **/
  bool hasDisagreements() const;

  /**
   * Get a list of tensors over which there are disagreements.
   **/
  const Disagreements &getDisagreements() const;

  /**
   * Clear both change and disagreement flags as well as the list of
   * disagreements.
   **/
  void clearChanges();

  /**
   * Set graph schedules (needed before this object is used).
   **/
  void setGraphSchedules(const GraphSchedules &graphSchedules);

  /**
   * Friend declaration for string operator.
   **/
  friend std::ostream &operator<<(std::ostream &out,
                                  const ReplicaEqualAnalysisResults &results);

private:
  // True if there have been changes since the last
  // `clearChanges` call.
  bool haveChange;
  // True if there have been disagreeing values since the last
  // `clearChanges` call.
  bool haveDisagreement;

  // Actual result map.
  Results results;
  // Look up the time associated with an Op.
  OpTimeMap opTimeMap;
  // Cache of graph schedules.
  GraphSchedules graphSchedules;
  // Tensors on which there are disagreements.
  Disagreements disagreements;

  // Time value used for initialisation time.
  static const Time initTime;
};

/**
 * Output results to stream (for diagnostics).
 *
 * \param out Output stream.
 * \return std::ostream& Output stream.
 */
std::ostream &operator<<(std::ostream &out,
                         const ReplicaEqualAnalysisResults &results);

} // namespace popart

#endif
