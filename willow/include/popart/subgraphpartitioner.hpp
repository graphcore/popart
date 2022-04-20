// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPH_PARTITIONER_HPP
#define GUARD_NEURALNET_SUBGRAPH_PARTITIONER_HPP

#include <cstddef>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <popart/names.hpp>

namespace popart {

class Op;
class CallOp;
class Graph;
class Ir;

namespace liveness {

class LivenessAnalyzer;

/**
 * When lowering CallOps, we would previously copy all tensors from the call
 * site (the CallOp's input tensors) to the subgraph's input tensors, do the
 * call and then copy the subgraph's output tensors back to the call site's
 * output tensors:
 *
 *   Copy(caller_in_1, subgraph_in_1)
 *   Copy(caller_in_2, subgraph_in_2)
 *   Call(subgraph)
 *   Copy(subgraph_out_1, caller_out_1)
 *   Copy(subgraph_out_2, caller_out_2)
 *
 * With this approach both, subgraph_in_1 and subgraph_in_2 are live during
 * the call. This can be suboptimal -- in some cases some subgraph inputs may
 * not be required until later in the subgraph and copying them later would
 * improve the required memory. Analogously, in some cases some subgraph outputs
 * may be ready to copy well before the end of the subgraph and it may be
 * advantageous to do this copy early. This is especially true for subgraphs
 * that deal with multiple inputs/outputs in sequence.
 *
 * To that end, graphs now support lowering over multiple "subgraph parts" to
 * allow CallOps that have these subgraphs as their called graph to copy inputs
 * later and outputs earlier. Essentially, each graph is 'split' over
 * multiple PopART fragments / Poplar sequences to facilitate any parent graph
 * that calls it to do a Copy of inputs or outputs between.
 *
 * The scheduling of copies for subgraph ops is already modelled by the
 * LivenessAnalyzer. We base our partitioning on this model. This class simply
 * interprets the LivenessAnalyzer's schedule and determines how to split
 * subgraphs into parts based on the LivenessAnalyzer's schedule.
 **/
class SubgraphPartitioner {
public:
  /**
   * Enum type for CallOpPart types.
   */
  enum class CallOpPartType {
    Undefined = 0,
    CopyInput,
    CopyOutput,
    CopyModified,
    CallSubgraphPart
  };

  /**
   * A class to represent a part of a CallOp.
   */
  class CallOpPart {
  public:
    // Type enum.
    CallOpPartType type;
    // Index of the input tensor (when type is a CopyInput/CopyModified).
    InIndex inIndex;
    // Index of the output tensor (when type is a CopyOutput).
    OutIndex outIndex;
    // Index of the called subgraph part (when type is CallSubgraphPart).
    SubgraphPartIndex subgraphPartIndex;
  };

  // Shorthand.
  using CallOpSchedule = std::vector<std::tuple<CallOpPart, SubgraphPartIndex>>;

  /**
   * Default contructor.
   */
  SubgraphPartitioner() = default;

  /**
   * Default destructor.
   */
  virtual ~SubgraphPartitioner() = default;

  /**
   * Prepare the results. Errors if IR or liveness analyser not set.
   */
  virtual void apply();

  /**
   * Set the IR dependency to use.
   */
  virtual void setIr(const Ir *);

  /**
   * Set the LivenessAnalyzer dependency to use.
   */
  virtual void setLivenessAnalyzer(const LivenessAnalyzer *);

  /**
   * Interpret the liveness analysis and work out what how many subgraph parts
   * a graph needs to lower all fragments between input/output copies.
   * Errors if apply was not run.
   */
  virtual int getNumSubgraphParts(const Graph &) const;

  /**
   * Interpret the liveness analysis and work out what subgraph part an
   * op is in based on the copying of inputs/outputs the subgraph the
   * op is in. For ops that spread over multiple subgraph parts (i.e. CallOps)
   * this returns the first such part. Errors if apply was not run.
   */
  virtual SubgraphPartIndex getOpSubgraphPartBegin(Op *) const;

  /**
   * Interpret the liveness analysis and work out what index is one larger
   * than the last subgraph part an op is in based on the copying of
   * inputs/outputs the subgraph the op is in. For ops that spread over
   * multiple subgraph parts (i.e. CallOps) this returns the last such part.
   * Errors if apply was not run.
   */
  virtual SubgraphPartIndex getOpSubgraphPartEnd(Op *) const;

  /**
   * Intepret the liveness analysis results and work out how a CallOp is
   * broken down over various subgraph parts. The result is a vector of pairs
   * of CallOp 'parts' and the 'subgraph parts' they should be lowered in.
   */
  virtual CallOpSchedule getCallOpSchedule(CallOp *) const;

  /**
   * Returns true for a graph if we support it being 'broken' into multiple
   * subgraph parts. The main graph does not support this. Subgraphs that are
   * called by any op that is not a CallOp also do not support this.
   */
  static bool isPartitionable(const Graph &graph);

private:
  // Class for internal use.
  class Node {
  public:
    // The op associated with this node.
    Op *op;
    // If op is a callop, this specifies which part of the call op the node
    // encapsulates. E.g. copy of an input, copy of an output or a call to a
    // subgraph part.
    CallOpPart callOpPart;
  };

  // Sequence of events and a list of indexes for subgraph part boundaries.
  // We use this for intermediate representations because it's easier to merge
  // and convert to SubgraphPartitions before adding to cache.
  using SubgraphPartitionTmp = std::tuple<std::vector<Node>, std::set<size_t>>;
  // Sequence of events mapped to the subgraph part in which they are lowered.
  using SubgraphPartition = std::vector<std::tuple<Node, SubgraphPartIndex>>;

  // Top-level function to work out a subgraph's lowering schedule.
  SubgraphPartition determineSubgraphPartition(const Graph &graph,
                                               bool partitionable);
  // Determine the order of things that are lowered (for specific instance).
  SubgraphPartitionTmp
  getSubgraphPartitionForInstance(const Graph &graph,
                                  const std::vector<Op *> &schedule,
                                  size_t enter,
                                  size_t exit,
                                  size_t callstackSize);
  // Check sequence of events matches and combine subgraph partition boundaries.
  SubgraphPartitionTmp mergeSubgraphPartitions(const Graph &graph,
                                               const SubgraphPartitionTmp &,
                                               const SubgraphPartitionTmp &);
  // Turn subgraph partition into final representation.
  SubgraphPartition finaliseSubgraphPartition(const SubgraphPartitionTmp &);

  // Helper function.
  void populateCache(const Graph &graph);
  // Helper function.
  void populateCacheForCalledGraphs(const Graph &graph);

  // Helper function.
  void logSubgraphPartitionTmp(const Graph &graph,
                               const SubgraphPartitionTmp &) const;
  // Helper function.
  void logSubgraphPartition(const Graph &graph,
                            const SubgraphPartition &,
                            bool partitionable) const;

  // Ir instance (dependency).
  const Ir *ir;
  // LivenessAnalyzer instance (dependency).
  const LivenessAnalyzer *liveness;

  // Op* to subgraphPart mapping cache per Graph.
  std::map<std::string, SubgraphPartition> cache;

  // Friend declarations for stream functions.
  friend std::ostream &operator<<(std::ostream &os, const Node &);
};

// Stream methods.
std::ostream &operator<<(std::ostream &os,
                         const SubgraphPartitioner::CallOpPartType &);
std::ostream &operator<<(std::ostream &os,
                         const SubgraphPartitioner::CallOpPart &);
std::ostream &operator<<(std::ostream &os, const SubgraphPartitioner::Node &);

} // namespace liveness
} // namespace popart

#endif
