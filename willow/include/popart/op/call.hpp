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

  const Graph &getCalledGraph() const;

  void appendAttributes(OpSerialiserBase &os) const override;

  bool isInputModified(InIndex);

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::vector<const Graph *> getCalledGraphs() const override;

  std::vector<TensorId> getInputsForGraph(const Graph &) const override;

private:
  std::reference_wrapper<Graph> callee;
};

} // namespace popart

#endif
