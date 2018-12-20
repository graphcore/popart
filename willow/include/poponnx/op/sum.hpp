#ifndef GUARD_NEURALNET_SUM_HPP
#define GUARD_NEURALNET_SUM_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class SumOp : public Op {
public:
  SumOp(const OperatorIdentifier &_opid,
        Ir *_ir,
        const std::string &name = "",
        const Attributes &_attr = {});
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  // The sum can have an variable number if input tensors, so can not define
  // the fixed input index's

  static OutIndex getOutIndex() { return 0; }
};
} // namespace poponnx

#endif
