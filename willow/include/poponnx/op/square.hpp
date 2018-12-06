#ifndef GUARD_NEURALNET_SQUARE_HPP
#define GUARD_NEURALNET_SQUARE_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class SquareOp : public ElementWiseUnaryOp {
public:
  SquareOp(const OpConstructorBundle &);
  SquareOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace poponnx

#endif
