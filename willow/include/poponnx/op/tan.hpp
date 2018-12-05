#ifndef GUARD_NEURALNET_TAN_HPP
#define GUARD_NEURALNET_TAN_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class TanOp : public Op {
public:
  TanOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
