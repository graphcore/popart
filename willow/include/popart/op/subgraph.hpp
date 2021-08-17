// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPHOP_HPP
#define GUARD_NEURALNET_SUBGRAPHOP_HPP

#include <popart/op.hpp>
#include <popart/transforms/autodiff/calledgraphgradophelper.hpp>

namespace popart {

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

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  virtual InIndex subgraphInToOpInIndex(InIndex index) const = 0;
  virtual InIndex opInToSubgraphInIndex(InIndex index) const = 0;

  virtual OutIndex subgraphOutToOpOutIndex(OutIndex index) const = 0;
  virtual OutIndex opOutToSubgraphOutIndex(OutIndex index) const = 0;

  virtual Graph &getCalledGraph() const = 0;
  virtual std::vector<const Graph *> getCalledGraphs() const override;
  virtual void setCalledGraph(Graph &) = 0;

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex index,
                                   std::set<OpId> &visited) const override;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex index,
                                    std::set<OpId> &visited) const override;

  bool hasSideEffect() const override;

  virtual InIndex opInToSubgraphInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex) override;
  virtual InIndex subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex) override;
  virtual OutIndex opOutToSubgraphOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex) override;
  virtual OutIndex subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex) override;

  // Override to avoid getGradOps being called before we're ready.
  virtual float calcAutoVirtualGraphCost(std::set<int> &inputs_seen) override;

  // Pass on to `calledGraphGradOpHelper`
  virtual void setCalledSubgraphGradInfo(
      const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) override;

protected:
  CalledGraphGradOpHelper calledGraphGradOpHelper;

private:
  // Regions of Input Tensors (InIndex) are aliased by Output Tensors (OutIndex)
  std::map<std::pair<InIndex, OutIndex>, std::pair<view::Chains, view::Chains>>
      aliasMap;
  std::map<InIndex, view::Regions> modifiesMap;
};

} // namespace popart

#endif
