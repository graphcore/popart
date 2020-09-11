// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CALL_HPP
#define GUARD_NEURALNET_CALL_HPP

#include <popart/op.hpp>
#include <popart/op/subgraphop.hpp>

namespace popart {

class CallOp : public SubgraphOp {
public:
  // parent: Graph this CallOp belongs to
  // callee: Graph this CallOp executes
  CallOp(const OperatorIdentifier &, Graph &parent, Graph &callee);

  void setup() final;
  std::unique_ptr<Op> clone() const final;

  Graph &getCalledGraph() const;

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::vector<TensorId> getGradOpInputIds(const Graph &gradGraph);

  void appendOutlineAttributes(OpSerialiserBase &os) const override;

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex index) const override;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex index) const override;

  std::vector<const Graph *> getCalledGraphs() const override;

  std::vector<TensorId> getInputsForGraph(const Graph &) const override;

  void addAlias(InIndex in,
                OutIndex out,
                view::Chains fwdChains,
                view::Chains bwdChains);

  void addModified(InIndex in, view::Regions regions);

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  GraphId getBackwardsGraphId() const;

private:
  std::reference_wrapper<Graph> callee;
  // Regions of Input Tensors (InIndex) are aliased by Output Tensors (OutIndex)
  std::map<std::pair<InIndex, OutIndex>, std::pair<view::Chains, view::Chains>>
      aliasMap;
  std::map<InIndex, view::Regions> modifiesMap;

  std::vector<GradInOutMapper>
  getGradInInfo(const std::vector<TensorId> &gradOpInputIds) const;
};

class CallGradOp : public CallOp {
public:
  CallGradOp(CallOp &fwdOp, const std::vector<GradInOutMapper> &gradInInfo_);

  // std::unique_ptr<Op> clone() const override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  std::vector<GradInOutMapper> gradInInfo;
  std::map<int, int> outInfoMap;
};

} // namespace popart

#endif
