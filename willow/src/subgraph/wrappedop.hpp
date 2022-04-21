// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPH_WRAPPED_OP_HPP
#define GUARD_NEURALNET_SUBGRAPH_WRAPPED_OP_HPP

#include <map>
#include <set>
#include <string>
#include <vector>

#include <popart/subgraph/subgraphnames.hpp>

#include <popart/analysis/replicaequal/replicaequalanalysis.hpp>

namespace popart {
// Forward declarations.
class Op;
class Ir;
class ReplicaEqualAnalysis;
} // namespace popart

namespace fwtools {
namespace subgraph {

/**
 * Class that wraps an op so that the subgraph outlining code can call
 * ```
 * wrappedOp->getSubgraphEquivId();
 * ```
 * Under the hood we call:
 * ```
 * op->getSubgraphEquivId(reAnalysis.getOpAttrs(op));
 * ```
 * where `reAnalysis` is a `ReplicaEqualAnalysis` object. We wrap ops in this
 * way to avoid changing the outlining code.
 */
class WrappedOp {
public:
  /**
   * Construct a WrappedOp.
   */
  WrappedOp(const popart::Op *op,
            const popart::ReplicaEqualAnalysis &reAnalysis);

  /**
   * Set mapping from Op* to WrappedOp* so we can implement getSubgraphInputs()
   * and getSubgraphOutputs(). This function must be called before using the
   * WrappedOp.
   */
  void setWrapMap(const std::map<popart::Op *, WrappedOp *> &wrapMap);

  /**
   * Wraps Op::getSubgraphInputs().
   */
  std::map<fwtools::subgraph::InIndex,
           std::tuple<WrappedOp *, fwtools::subgraph::OutIndex, std::string>>
  getSubgraphInputs() const;

  /**
   * Wraps Op::getSubgraphOutputs().
   */
  std::map<fwtools::subgraph::OutIndex, std::set<WrappedOp *>>
  getSubgraphOutputs() const;

  /**
   * Wraps Op::getSubgraphValue().
   */
  float getSubgraphValue() const;

  /**
   * Get the popart::Op wrapped in this object.
   * \return The op.
   */
  const popart::Op *unwrap() const;

  /**
   * Get the popart::ReplicaEqualAnalysis reference wrapped in this object.
   * \return The replica equal analysis.
   */
  const popart::ReplicaEqualAnalysis &getReplicaEqualAnalysis() const;

  /**
   * Call `getSubgraphEquivId` on the wrapped op, passing in the
   * Op-attributes from the ReplicaEqualAnalysis member.
   * \return The equivalence class representive.
   */
  std::string getSubgraphEquivId();

private:
  // Lookop Op* to WrappedOp* mapping using wrapMap.
  WrappedOp *getWrappedOp(popart::Op *op) const;

  // The op we are wrapping.
  const popart::Op *op;

  // A reference to a ReplicaEqual object.
  std::reference_wrapper<const popart::ReplicaEqualAnalysis> reAnalysis;

  // Mapping from Op* to WrappedOp*.
  std::map<popart::Op *, WrappedOp *> wrapMap;
};

/**
 * Result object for toWrappedOpSched function.
 */
struct WrappedOpSched {
  /**
   * Schedule as vector of raw pointers.
   */
  std::vector<WrappedOp *> rawPtrs;

  /**
   * Schedule as vector of shared pointers.
   */
  std::vector<std::shared_ptr<WrappedOp>> sharedPtrs;

  /**
   * Raw pointer to wrapped placeholder op.
   */
  WrappedOp *rawPlaceholderPtr;

  /**
   * Shared pointer to wrapped placeholder op.
   */
  std::shared_ptr<WrappedOp> sharedPlaceholderPtr;
};

/**
 * A convenience function to convert a std::vector of Op*s to a std::vector of
 * WrappedOp*s (which is what the outliner code needs) with an additional vector
 * being returned with smart pointers that should be used to deallocate the
 * WrappedOps.
 *
 * \param reAnalysis A reference to a ReplicaEqualAnalysis object.
 * \param sched An Op schedule.
 * \return A schedule of WrappedOps (both raw and shared pointers).
 */
WrappedOpSched toWrappedOpSched(popart::Ir &ir,
                                const popart::ReplicaEqualAnalysis &reAnalysis,
                                const std::vector<popart::Op *> &sched);

} // namespace subgraph
} // namespace fwtools

#endif
