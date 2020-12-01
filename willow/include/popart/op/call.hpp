// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CALL_HPP
#define GUARD_NEURALNET_CALL_HPP

#include <popart/op.hpp>
#include <popart/op/subgraph.hpp>

namespace popart {

class CallOp : public SubgraphOp {
public:
  // parent: Graph this CallOp belongs to
  // callee: Graph this CallOp executes
  CallOp(const OperatorIdentifier &, Graph &parent, Graph &callee);

  void setup() final;
  std::unique_ptr<Op> clone() const final;

  Graph &getCalledGraph() const override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::vector<TensorId> getGradOpInputIds(const Graph &gradGraph);

  void appendOutlineAttributes(OpSerialiserBase &os) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::vector<const Graph *> getCalledGraphs() const override;

  GraphId getBackwardsGraphId() const;

  InIndex subgraphInToOpInIndex(InIndex index) const override { return index; }
  InIndex opInToSubgraphInIndex(InIndex index) const override { return index; }
  OutIndex subgraphOutToOpOutIndex(OutIndex index) const override {
    return index;
  }
  OutIndex opOutToSubgraphOutIndex(OutIndex index) const override {
    return index;
  }

private:
  std::reference_wrapper<Graph> callee;

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
