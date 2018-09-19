#ifndef GUARD_NEURALNET_RELU_HPP
#define GUARD_NEURALNET_RELU_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class ReluOp : public Op {
public:
  ReluOp(const onnx::NodeProto &node, Graph *pgraph);

  virtual void setup() override final;
};
} // namespace neuralnet

#endif
