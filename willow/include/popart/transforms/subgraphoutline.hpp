// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPHOUTLINE_HPP
#define GUARD_NEURALNET_SUBGRAPHOUTLINE_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

// Forward declaration.
class AliasesMap;

/**
 * Class describing a group cluster of Ops belonging to the same Graph
 * that can be replaced be replaced by a call to a single subgraph.
 **/
class SubgraphableOpCluster {
public:
  SubgraphableOpCluster(const std::vector<OpId> &, Graph *);

  std::vector<OpId> ops;
  Graph *graph;

  std::vector<Tensor *> external_inputs;
  std::vector<Tensor *> external_outputs;
  std::set<Tensor *> all_outputs;

  int getIndex(const Op *) const;

  Graph &getGraph() const { return *graph; }

private:
  void addExternalOutput(Tensor *);
  void addExternalInput(Tensor *);
};

/**
 * Class for creating functionally equivalent subgraphs from
 * SubgraphableOpClusters, and replacing instances of
 * SubgraphableOpClusters with calls to these subgraphs. Further down the stack,
 * this allows for code-reuse, which results in a lower memory footprint
 * for the compiled graph.
 **/
class SubgraphOutline : public Transform {
public:
  static std::size_t id();

  SubgraphOutline() : Transform() {}
  virtual ~SubgraphOutline() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "SubgraphOutline"; }

  /**
   * Create a subgraph from a set of identitcal op clusters.
   *
   * \param instances A set of SubgraphableOpClusters that can be replaced by a
                      call to the same subgraph. All SubgraphableOpCluster
                      instances must be functionally equivalent.
   * \param ir The IR.
   * \param index_map An empty map, passed by reference. Used to map from ops in
   *                  the new subgraph to their corresponding indices in the
   *                  first SubgraphableOpCluster instance. Required as input
   *                  argument to 'replaceWithCallOp'.
   * \param subgraphId The returned subgraph's id.
   * \return A Graph that is functionally equivalent to each
   *         SubgraphableOpCluster instance.
   **/
  static Graph &
  createSubgraph(const std::vector<SubgraphableOpCluster> instances,
                 Ir &ir,
                 std::map<Op *, int> &index_map,
                 std::string subgraphId = "call");

  /**
   * Create an 'empty' subgraph from an op cluster.
   *
   * \param instance A SubgraphableOpCluster that is used as a template
   *                 for which we build 'empty' subgraph where
   *                 inputs and output tensors can be connected via
   *                 nops and output tensors can be set to default values.
   * \param ir The IR.
   * \param index_map An empty map, passed by reference. Used to map from op in
   *                  the new subgraph to their corresponding indices in the
   *                  SubgraphableOpCluster instance. Required as input
   *                  argument to 'replaceWithEmptyElseBranchIfOp'.
   * \param subgraphId The returned subgraph's id.
   * \param identityInputToOutputIndiciesMapping Specifies the connections
   *  of inputs to outputs via nop operations in the 'empty' subgraph.
   *  Each pair must have the same shape and type.
   * \param outputIndiciesAndValues Map of pairs of output indices and values.
   * \return A Graph, low compute subgraph which stands for the op when
   *                  it is not executed.
   **/
  static Graph &createEmptySubgraph(
      const SubgraphableOpCluster &instance,
      Ir &ir,
      std::map<Op *, int> &index_map,
      std::string subgraphId,
      const std::map<InIndex, OutIndex> &identityInputToOutputIndiciesMapping,
      const std::map<OutIndex, float> &outputIndiciesAndValues);

  // Helper function to reuse code.
  static void setSubgraphOpSettingsFromClusterInstance(
      Op *op,
      const SubgraphableOpCluster &instance);

  /**
   * Replace a cluster of ops with a call to a subgraph.
   *
   * \param instance The SubgraphableOpClusters instance to be replaced.
   * \param subgraph The subgraph, a call to which is to replace the `instance`.
   * \param index_map Used to map from ops in the new subgraph to their
   *                  corresponding indices in the first SubgraphableOpCluster
   *                  instance.
   * \param aliasesMap AliasesMap with alias information for instance's graph.
   * \return The replacement CallOp's pointer.
   **/
  static Op *replaceWithCallOp(const SubgraphableOpCluster &instance,
                               Graph &subgraph,
                               const std::map<Op *, int> &index_map,
                               AliasesMap &aliasesMap);

  /**
   * Replace an op with if op. Where the op is moved to the first branch
   * of if op. Its second branch is for low intensity compute which passes
   * input tensors to outputs or provide default output tensors.
   *
   * \param instance The SubgraphableOpClusters instance which holds op
   * to be replaced.
   * \param subgraph if then branch subgraph which contains the op.
   * \param emptySubgraph if else low intensity compute branch subgraph.
   * \param index_map Used to map from ops in the new subgraph to their
   *                  corresponding indices in the first SubgraphableOpCluster
   *                  instance.
   * \param aliasesMap AliasesMap with alias information for instance's graph.
   * \param flag a Tensor deciding which branch should be used.
   * \return The replacement IfOp's pointer.
   **/
  static Op *
  replaceWithEmptyElseBranchIfOp(const SubgraphableOpCluster &instance,
                                 Graph &subgraph,
                                 Graph &emptySubgraph,
                                 const std::map<Op *, int> &index_map,
                                 AliasesMap &aliasesMap,
                                 Tensor *flag);
};

} // namespace popart

#endif
