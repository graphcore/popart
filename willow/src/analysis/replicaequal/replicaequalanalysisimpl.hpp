// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_ANALYSIS_REPLICAEQUAL_REPLICAEQUALANALYSISIMPL_HPP_
#define POPART_WILLOW_SRC_ANALYSIS_REPLICAEQUAL_REPLICAEQUALANALYSISIMPL_HPP_

#include <analysis/replicaequal/replicaequalanalysisresults.hpp>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <popart/analysis/replicaequal/replicaequalanalysisproxy.hpp>

#include "popart/alias/aliasmodel.hpp"
#include "popart/graphid.hpp"
#include "popart/names.hpp"

namespace popart {
class Graph;
class Ir;
class Op;
class Tensor;
class any;

/**
 * Implementation of ReplicaEqualAnalysis.
 **/
class ReplicaEqualAnalysisImpl : public ReplicaEqualAnalysisProxy {
public:
  ReplicaEqualAnalysisImpl(const Ir &ir_);
  ReplicaEqualAnalysisImpl(const Ir &ir_, AliasModel &aliasModel_);

  /**
   * See ReplicaEqualAnalysis.
   **/
  void apply();

  /**
   * See ReplicaEqualAnalysis.
   **/
  virtual IsReplicaEqual isOpInputEqual(const Op *op, InIndex inIndex) const;

  /**
   * See ReplicaEqualAnalysis.
   **/
  virtual IsReplicaEqual isOpOutputEqual(const Op *op, OutIndex outIndex) const;

  /**
   * See ReplicaEqualAnalysis.
   **/
  virtual std::map<std::string, popart::any> getOpAttrs(const Op *op) const;

  /**
   * See ReplicaEqualAnalysisProxy.
   **/
  virtual ReplEqModifiedInputMap getModifiedInputMapFromAliases(
      const Op *op,
      const ReplEqOutputMap &replEqOpOutputMap) const override;

  /**
   * See ReplicaEqualAnalysisProxy.
   **/
  virtual std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqualThroughGraph(
      const Graph *graph,
      const ReplEqInputMap &replEqGraphInputMap) override;

private:
  // Shorthand for typenames.
  using Results        = ReplicaEqualAnalysisResults;
  using GraphSchedules = std::map<GraphId, std::vector<Op *>>;
  using Updates        = std::vector<std::tuple<Tensor *, IsReplicaEqual>>;

  /**
   * Initialise aliasmap and graph schedule cache.
   **/
  void initialise();

  /**
   * Add initial replica-equal values for variables tensors.
   **/
  void addVariableTensorsToAnalysisResults();

  /**
   * Add initial replica-equal values for streams tensors.
   **/
  void addStreamTensorsToAnalysisResults();

  /**
   * Add initial replica-equal values for const tensors.
   **/
  void addConstTensorsToAnalysisResults(const Graph *graph);

  /**
   * Add replica equal values for Op outputs to results. Any actual result
   * updates are appended to the updates parameter.
   *
   * \param op The Op to add the replica-equal values to results for.
   * \param isReplEqMap The replica-equal values to add.
   * \param updates List of updated tensor values.
   **/
  void addOpOutputValsToAnalysisResults(const Op *op,
                                        const ReplEqOutputMap &isReplEqMap,
                                        Updates &updates);

  /**
   * Add replica equal values for Op's modified inputs to results. Any actual
   * result updates are appended to the updates parameter.
   *
   * \param op The Op to add the replica-equal values to results for.
   * \param isReplEqMap The replica-equal values to add.
   * \param updates List of updated tensor values.
   **/
  void addOpModifiedInputValsToAnalysisResults(
      const Op *op,
      const ReplEqModifiedInputMap &isReplEqMap,
      Updates &updates);

  /**
   * Add replica equal values of graph inputs to results.
   *
   * \param graph The graph to populate inputs of.
   * \param replEqInputMap A mapping from input indices to replica-equal values.
   * \return True if the analysis results changed as a result of this call.
   **/
  bool addGraphInputValsToAnalysisResults(const Graph *graph,
                                          const ReplEqInputMap &inputMap);

  /**
   * Add replica equal values for aliases of updated tensors.
   *
   * \param op The Op to add the replica-equal values to results for.
   * \param updates The replica-equal values to add.
   **/
  void addAliasesValsToAnalysisResults(const Op *op, const Updates &updates);

  /**
   * Get replica equal mapping for Op inputs from a Results
   * and put it in ReplEqInputMap.
   *
   * \param op The Op to get replica-equal values of inputs for.
   * \return A ReplEqInputMap with replica-equal values for the Op's
   *    input tensors.
   **/
  ReplEqInputMap getOpInputValsFromAnalysisResults(const Op *op) const;

  /**
   * Get replica equal value of graph outputs from a Results and put them
   * in a ReplEqOutputMap.
   *
   * \param graph The graph to populate inputs of.
   * \return A ReplEqOutputMap with replica equal values for the
   *    graph's outputs.
   **/
  ReplEqOutputMap
  getGraphOutputValsFromAnalysisResults(const Graph *graph) const;

  /**
   * Get replica equal value of graph inputs from a Results and put them
   * in a ReplEqModifiedInputMap.
   *
   * \param graph The graph to populate inputs of.
   * \return A ReplEqModifiedInputMap with the (final) replica equal values for
   *the graph's inputs.
   **/
  ReplEqModifiedInputMap
  getGraphInputValsFromAnalysisResults(const Graph *graph) const;

  /**
   * If a tensor is not replica-equal and aliases a variable tensor in
   * the main graph, then the replica-equal value of the variable should be
   * adjusted, also. Note this only looks at changed values.
   * \param results The result set to process.
   **/
  void processMainGraphAliases();

  // Member alias map (used ONLY if no AliasModel is passed via the
  // constructor).
  AliasModel localAliasModel;
  // Reference to intermediate representation.
  std::reference_wrapper<const Ir> ir;
  // References either the AliasModel from the constructor OR localAliasModel.
  std::reference_wrapper<AliasModel> aliasModel;
  // Analysis result.
  Results analysisResults;
  // Cache containing non-optimal graph schedules. Iterating over this member
  // must be avoided because the order of an unordered map is not deterministic.
  GraphSchedules graphSchedules;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_ANALYSIS_REPLICAEQUAL_REPLICAEQUALANALYSISIMPL_HPP_
