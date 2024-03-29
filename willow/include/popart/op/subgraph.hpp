// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SUBGRAPH_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SUBGRAPH_HPP_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/chains.hpp>
#include <popart/op.hpp>
#include <popart/transforms/autodiff/calledgraphgradophelper.hpp>

#include "popart/bwdgraphinfo.hpp"
#include "popart/names.hpp"
#include "popart/tensorlocation.hpp"

namespace onnx {
class GraphProto;
} // namespace onnx

namespace popart {
class AliasModel;
class Graph;
class OpSerialiserBase;
class ReplicaEqualAnalysisProxy;
class TensorInfo;
struct OperatorIdentifier;

class SubgraphOp : public Op {
public:
  static bool existsInBodyInputs(std::vector<std::string> &loopBodyInputIds,
                                 TensorId &tensorId);

  static bool
  existsInOpInputs(std::vector<std::pair<TensorId, TensorInfo>> &opInputs,
                   TensorId &tensorId);

  static std::vector<TensorId>
  getBodyInputIds(const ONNX_NAMESPACE::GraphProto &bodyProto);

  static std::vector<TensorId>
  getBodyOutputIds(const ONNX_NAMESPACE::GraphProto &bodyProto);

  // parent: Graph this CallOp belongs to
  SubgraphOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override = 0;

  void appendOutlineAttributes(OpSerialiserBase &os) const override;

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;
  void addAlias(InIndex in,
                OutIndex out,
                view::Chains fwdChains,
                view::Chains bwdChains);

  void adjustAliasInIndices(InIndex fromIn, InIndex toIn);
  void adjustAliasOutIndices(OutIndex fromOut, OutIndex toOut);
  void adjustModifiedIndices(InIndex fromIn, InIndex toIn);

  void addModified(InIndex in, view::Regions regions);
  void removeModified(InIndex in);
  void removeAlias(InIndex in, OutIndex out);

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const override;

  virtual InIndex subgraphInToOpInIndex(InIndex index) const = 0;
  virtual InIndex opInToSubgraphInIndex(InIndex index) const = 0;

  virtual OutIndex subgraphOutToOpOutIndex(OutIndex index) const = 0;
  virtual OutIndex opOutToSubgraphOutIndex(OutIndex index) const = 0;

  virtual Graph &getCalledGraph() const = 0;
  std::vector<const Graph *> getCalledGraphs() const override;
  virtual void setCalledGraph(Graph &) = 0;

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex index,
                                   std::set<OpId> &visited) const override;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex index,
                                    std::set<OpId> &visited) const override;

  bool hasSideEffect() const override;

  virtual InIndex opInToSubgraphInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex) const override;
  virtual InIndex subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex) const override;
  virtual OutIndex opOutToSubgraphOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex) const override;
  virtual OutIndex subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex) const override;

  // Override to avoid getGradOps being called before we're ready.
  float calcAutoVirtualGraphCost(std::set<int> &inputs_seen) override;

  // Pass on to `calledGraphGradOpHelper`
  virtual void setCalledSubgraphGradInfo(
      const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) override;

protected:
  CalledGraphGradOpHelper calledGraphGradOpHelper;
  // Regions of Input Tensors (InIndex) are aliased by Output Tensors (OutIndex)
  std::map<std::pair<InIndex, OutIndex>, std::pair<view::Chains, view::Chains>>
      aliasMap;
  std::map<InIndex, view::Regions> modifiesMap;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SUBGRAPH_HPP_
