// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef NEURALNET_ANALYSIS_REPLICA_EQUAL_ANALYSIS_PROXY_HPP
#define NEURALNET_ANALYSIS_REPLICA_EQUAL_ANALYSIS_PROXY_HPP

#include <popart/names.hpp>

namespace popart {

// Forward declaration.
class Op;

/**
 * Interface for object passed to `Op::fwdPropagateIsReplicaEqual`.
 **/
class ReplicaEqualAnalysisProxy {
public:
  /**
   * Work out replica-equal values for modified inputs by setting replica-equal
   * values of modified inputs to true if and only if the Op has an output that
   * is an alias of the modified input, containing all elements of the input,
   * and the output is deemed replica-equal. If this doesn't hold a modified
   * input is assumed to be not replica-equal.
   *
   * NOTE: It is possible for an Op to modify an input to a replica-equal value
   * in a way that will not be detected by this implementation, but it's
   * generally true for currently supported Ops at time of writing.
   *
   * \param op The Op to get the replica-equal values for modified inputs for.
   * \param replEqOpOutputMap The Op's replica-equal output values.
   * \return A mapping containing replica-equal values for modified outputs.
   **/
  virtual ReplEqModifiedInputMap getModifiedInputMapFromAliases(
      const Op *op,
      const ReplEqOutputMap &replEqOpOutputMap) const = 0;

  /**
   * A method that can be called to work out how replica-equal values for
   * graph inputs propagate to replica-equal values for graph outputs.
   *
   * NOTE: Graphs never copy-modify input tensors, although Ops that call
   * graphs might (like CallOp, LoopOp).
   *
   * \param graph The graph to propagate replica-equal values through.
   * \param replEqGraphInputMap The replica-equal values for the graph's inputs.
   * \return A tuple containing a ReplEqOutputMap that describes replica-equal
   *    values for the graph's outputs and a ReplEqModifiedInputMap that
   *    describes the final replica-equal values of the graph's inputs.
   **/
  virtual std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqualThroughGraph(
      const Graph *graph,
      const ReplEqInputMap &replEqGraphInputMap) = 0;

  virtual ~ReplicaEqualAnalysisProxy() {}
};

} // namespace popart

#endif
