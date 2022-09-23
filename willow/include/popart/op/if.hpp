// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_IF_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_IF_HPP_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <popart/graphid.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/transforms/autodiff/calledgraphgradophelper.hpp>

#include "popart/bwdgraphinfo.hpp"
#include "popart/names.hpp"

namespace popart {
class AliasModel;
class Graph;
class ReplicaEqualAnalysisProxy;
class Scope;
struct OperatorIdentifier;

struct BranchInfo {
  BranchInfo(const GraphId &,
             const std::map<int, int> inputIndicesMap,
             const std::map<int, int> outputIndicesMap);

  GraphId graphId;
  // IfOp input indices to Graph input indices.
  std::map<int, int> inputIndicesMap;
  // IfOp output indices to Graph output indices.
  std::map<int, int> outputIndicesMap;
};

class IfOp : public Op {
public:
  IfOp(const OperatorIdentifier &,
       const BranchInfo &thenBranchInfo,
       const BranchInfo &elseBranchInfo,
       const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getConditionInIndex() { return 0; }

  Graph &getThenGraph() const;
  Graph &getElseGraph() const;

  const std::map<InIndex, InIndex> &getBranchInIndicesMap(const Graph &) const;
  const std::map<OutIndex, OutIndex> &
  getBranchOutIndicesMap(const Graph &) const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const override;

  std::vector<const Graph *> getCalledGraphs() const override;

  virtual InIndex opInToSubgraphInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex) const override;
  virtual InIndex subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex) const override;
  virtual OutIndex opOutToSubgraphOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex) const override;
  virtual OutIndex subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex) const override;

  std::set<OutIndex> opInToOpOutIndex(InIndex in) const override;
  std::set<InIndex> opOutToOpInIndex(OutIndex out) const override;

  // Override to avoid getGradOps being called before we're ready.
  float calcAutoVirtualGraphCost(std::set<int> &inputs_seen) override;

  // Pass on to `calledGraphGradOpHelper`
  virtual void setCalledSubgraphGradInfo(
      const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) override;

private:
  void appendInputs(const std::vector<TensorId> &inputIds, const Scope &);

  // Map of branchOutput ids to corresponding op out ids
  std::map<TensorId, TensorId> getBranchOutIdToOpOutIdMap() const;
  // Get the input TensorIds for the grad op
  std::vector<TensorId> getGradOpInputIds(const Graph &gradThenGraph,
                                          const Graph &gradElseGraph);
  // for each bwdGraph input, get the corresponding TensorId in the IfOps scope
  std::map<TensorId, int>
  getOpInIdToBwdGraphInIndexMap(const Graph &fwdGraph,
                                const Graph &bwdGraph) const;
  std::vector<GradInOutMapper>
  getGradInInfo(const std::vector<TensorId> &gradOpInputIds) const;

  std::map<int, int> getInIndicesMapForGradOp(
      const std::map<TensorId, int> &opInIdToOpInIdx,
      const std::map<TensorId, int> &opInIdToGraphInIdx) const;
  std::map<int, int>
  getOutIndicesMapForGradOp(const std::map<InIndex, InIndex> &idxMap) const;

  BranchInfo
  getBwdGraphBranchInfo(const Graph &fwdGraph,
                        const Graph &bwdGraph,
                        const std::vector<TensorId> &gradOpInputIds) const;

  const std::vector<TensorId> inputIds;

  // opIndex -> branchIndex
  const std::map<InIndex, InIndex> thenInputIndicesMap;
  const std::map<InIndex, InIndex> elseInputIndicesMap;
  const std::map<InIndex, InIndex> thenOutputIndicesMap;
  const std::map<InIndex, InIndex> elseOutputIndicesMap;

  const GraphId thenGraphId;
  const GraphId elseGraphId;

  CalledGraphGradOpHelper calledGraphGradOpHelper;
};

class IfGradOp : public IfOp {
public:
  IfGradOp(const IfOp &,
           const std::vector<GradInOutMapper> &gradInInfo,
           const BranchInfo &thenBranchInfo,
           const BranchInfo &elseBranchInfo);

  std::unique_ptr<Op> clone() const override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  std::vector<GradInOutMapper> gradInInfo;
  std::map<int, int> outInfoMap;
};

class IfConditionGradOp : public IdentityOp {
public:
  IfConditionGradOp(const IfOp &);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_IF_HPP_
