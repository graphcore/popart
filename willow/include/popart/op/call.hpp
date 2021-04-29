// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CALL_HPP
#define GUARD_NEURALNET_CALL_HPP

#include <popart/op.hpp>
#include <popart/op/subgraph.hpp>

namespace popart {

class CallOp : public SubgraphOp {
public:
  // callee: Graph this CallOp executes
  // NOTE: modifiedInputsViaAttrs is currently only used for testing purposes.
  CallOp(const OperatorIdentifier &,
         Graph &callee,
         const Op::Settings &settings);

  // callee: Graph this CallOp executes
  // NOTE: modifiedInputsViaAttrs is currently only used for testing purposes.
  CallOp(const OperatorIdentifier &,
         Graph &callee,
         std::vector<int> modifiedInputsViaAttrs,
         const Op::Settings &settings);

  void setup() final;
  std::unique_ptr<Op> clone() const final;

  Graph &getCalledGraph() const override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::vector<TensorId> getGradOpInputIds(const Graph &gradGraph);

  void appendOutlineAttributes(OpSerialiserBase &os) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::vector<const Graph *> getCalledGraphs() const override;
  void setCalledGraph(Graph &) override;

  InIndex subgraphInToOpInIndex(InIndex index) const override { return index; }
  InIndex opInToSubgraphInIndex(InIndex index) const override { return index; }
  OutIndex subgraphOutToOpOutIndex(OutIndex index) const override {
    return index;
  }
  OutIndex opOutToSubgraphOutIndex(OutIndex index) const override {
    return index;
  }

  // TODO(T30050) fully support this, at the moment it assumes no aliasing
  // between inputs and outputs.
  virtual void growAliaser(PoprithmsAliaser &m) const override {
    growAliaserMulti(m);
  }

private:
  std::reference_wrapper<Graph> callee;
  // Facility to auto-mark inputs as 'modified' on construction by setting the
  // 'modifiedInputs' attribute. This is currently only used for testing
  // purposes.
  std::vector<int> modifiedInputsViaAttrs;

  std::vector<GradInOutMapper>
  getGradInInfo(const std::vector<TensorId> &gradOpInputIds) const;
};

class CallGradOp : public CallOp {
public:
  CallGradOp(CallOp &fwdOp,
             Graph &bwdGraph,
             const std::vector<GradInOutMapper> &gradInInfo_,
             const std::map<int, int> &gradOutInfo_);

  // std::unique_ptr<Op> clone() const override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  std::vector<GradInOutMapper> gradInInfo;
  std::map<int, int> outInfoMap;
};

} // namespace popart

#endif
