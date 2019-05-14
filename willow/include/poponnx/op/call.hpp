#ifndef GUARD_NEURALNET_CALL_HPP
#define GUARD_NEURALNET_CALL_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class CallOp : public Op {
public:
  // parent: Graph this CallOp belongs to
  // callee: Graph this CallOp executes
  CallOp(Graph &parent, Graph &callee);

  void setup() final;
  std::unique_ptr<Op> clone() const final;

  Graph &getCalledGraph();

  void appendAttributes(OpSerialiserBase &os) const override;

  bool isInputModified(InIndex);

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  std::reference_wrapper<Graph> callee;
};

} // namespace poponnx

#endif
