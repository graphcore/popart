// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef NEURALNET_ANALYSIS_REPLICA_EQUAL_ANALYSIS_HPP
#define NEURALNET_ANALYSIS_REPLICA_EQUAL_ANALYSIS_HPP

#include <map>
#include <memory>
#include <string>

#include "popart/names.hpp"

namespace popart {

// Forward declaration.
class Op;
class ReplicaEqualAnalysisImpl;
class AliasModel;
class Ir;
class any;

/**
 * This class determines for each tensor in the IR whether or not it's value
 * is always equal across replicas.
 *
 * ASSUMPTION: All const tensors except the streamed seed are the same
 * on every replica.
 *
 * ASSUMPTION: Stream tensors of type BROADCAST are the same on every
 * replica, other stream tensors are not.
 *
 * Usage:
 * ```
 * ReplicaAnalysis analysis;
 * analysis.apply(ir);
 *
 * // Check analysis results.
 * auto res = analysis.isEqual(tensor);
 * if (res == IsReplicaEqual::True || res == IsReplicaEqual::Maybe) {
 *   ..
 * }
 * ```
 **/
class ReplicaEqualAnalysis {
public:
  /**
   * Construct a new Replica Equal Analysis object.
   *
   * \param ir The IR object to analyse.
   **/
  ReplicaEqualAnalysis(const Ir &ir);

  /**
   * Construct a new Replica Equal Analysis object (using a precomputed alias
   * model).
   *
   * \param ir The IR object to analyse.
   * \param aliasModel Alias mappings for all graphs in the IR.
   **/
  ReplicaEqualAnalysis(const Ir &ir, AliasModel &aliasModel);
  virtual ~ReplicaEqualAnalysis();

  /**
   * Do the analysis.
   **/
  virtual void apply();

  /**
   * Once the analysis is complete, use this function to determine for a given
   * Op input whether the input tensor is equal across replicas or not.
   *
   * NOTE: We query for Op inputs/outputs (instead of, say, tensors without
   * being in the context of an Op) because querying on the level of tensors can
   * be ambiguous when an IR contains Ops that modify tensors.
   *
   * \param op The op.
   * \param inIndex The input index to query.
   * \return Whether the input tensor is equal across replication.
   */
  virtual IsReplicaEqual isOpInputEqual(const Op *op, InIndex inIndex) const;

  /**
   * Once the analysis is complete, use this function to determine for a given
   * Op output whether the output tensor is equal across replicas or not.
   *
   * NOTE: We query for Op inputs/outputs (instead of, say, tensors without
   * being in the context of an Op) because querying on the level of tensors can
   * be ambiguous when an IR contains Ops that modify tensors.
   *
   * \param op The op.
   * \param outIndex The input index to query.
   * \return Whether the input tensor is equal across replication.
   */
  virtual IsReplicaEqual isOpOutputEqual(const Op *op, OutIndex outIndex) const;

  /**
   * Once the analysis is complete, use this function to get a set of attributes
   * that describe the replica-equalness of all Op inputs and outputs.
   *
   * \param op The op.
   * \return A map of attributes describing the Op's replica-equalness.
   */
  virtual std::map<std::string, popart::any> getOpAttrs(const Op *op) const;

private:
  // Hide implementation (PIMPL).
  std::unique_ptr<ReplicaEqualAnalysisImpl> impl;
};

} // namespace popart

#endif
