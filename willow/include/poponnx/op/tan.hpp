#ifndef GUARD_NEURALNET_TAN_HPP
#define GUARD_NEURALNET_TAN_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class TanOp : public ElementWiseUnaryOp {
public:
  TanOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace poponnx

#endif
