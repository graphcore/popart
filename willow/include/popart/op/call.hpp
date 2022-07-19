// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_CALL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_CALL_HPP_

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/subgraph.hpp>

#include "popart/names.hpp"

namespace popart {
class AliasModel;
class Graph;
class OpSerialiserBase;
struct OperatorIdentifier;

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
         const std::vector<int> &modifiedInputsViaAttrs,
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
  virtual std::set<OutIndex> opInToOpOutIndex(InIndex in) const override {
    return {};
  }
  virtual std::set<InIndex> opOutToOpInIndex(OutIndex out) const override {
    return {};
  }

  virtual void growAliasModel(AliasModel &m) const override {
    growAliasModelMulti(m);
  }

  void connectInTensor(InIndex inIndex, TensorId tenId) override;

private:
  std::reference_wrapper<Graph> callee;
  // Facility to auto-mark inputs as 'modified' on construction by setting the
  // 'modifiedInputs' attribute. This is currently only used for testing
  // purposes.
  const std::vector<int> modifiedInputsViaAttrs;

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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_CALL_HPP_
