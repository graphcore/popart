#ifndef GUARD_NEURALNET_SUM_HPP
#define GUARD_NEURALNET_SUM_HPP

#include <poponnx/ir.hpp>

namespace poponnx {

class SumOp : public Op {
public:
  SumOp(const OpConstructorBundle &);
  SumOp(const onnx::NodeProto &node, Ir *pir);
  void setup() final;
  std::unique_ptr<Op> clone() const final;
};
} // namespace poponnx

#endif
