#ifndef GUARD_NEURALNET_CALL_HPP
#define GUARD_NEURALNET_CALL_HPP

#include <popart/op.hpp>

namespace popart {

class CallOp : public Op {
public:
  // parent: Graph this CallOp belongs to
  // callee: Graph this CallOp executes
  CallOp(Graph &parent, Graph &callee);

  void setup() final;
  std::unique_ptr<Op> clone() const final;

  Graph &getCalledGraph() const;

  void appendOutlineAttributes(OpSerialiserBase &os) const override;

  bool isInputModified(InIndex);

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  VGraphId getIntrospectionInVirtualGraphId(InIndex index) const override;
  VGraphId getIntrospectionOutVirtualGraphId(OutIndex index) const override;

  std::vector<const Graph *> getCalledGraphs() const override;

  std::vector<TensorId> getInputsForGraph(const Graph &) const override;

  void addAlias(InIndex in,
                OutIndex out,
                view::Regions fwdRegions,
                view::Regions bwdRegions);

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

private:
  std::reference_wrapper<Graph> callee;
  // Regions of Input Tensors (InIndex) are aliased by Output Tensors (OutIndex)
  std::map<std::pair<InIndex, OutIndex>,
           std::pair<view::Regions, view::Regions>>
      aliasMap;
  std::map<InIndex, bool> inZeroCopy;
  std::map<OutIndex, bool> outZeroCopy;
};

} // namespace popart

#endif
