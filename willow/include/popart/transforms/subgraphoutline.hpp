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
   * Replace a cluster of ops with a call to a subgraph.
   *
   * \param instance The SubgraphableOpClusters instance to be replaced.
   * \param subgraph The subgraph, a call to which is to replace the `instance`.
   * \param index_map Used to map from ops in the new subgraph to their
   *                  corresponding indices in the first SubgraphableOpCluster
   *                  instance.
   * \param aliasesMap AliasesMap with alias information for instance's graph.
   * \param subgraphId The returned subgraph's id.
   * \return The replacement CallOp's pointer.
   **/
  static Op *replaceWithCallOp(const SubgraphableOpCluster &instance,
                               Graph &subgraph,
                               const std::map<Op *, int> &index_map,
                               AliasesMap &aliasesMap);
};

} // namespace popart

#endif
