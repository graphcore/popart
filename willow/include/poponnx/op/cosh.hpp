#ifndef GUARD_NEURALNET_COSH_HPP
#define GUARD_NEURALNET_COSH_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class CoshOp : public Op {
public:
  CoshOp(const OperatorIdentifier &_opid,
         Ir *_ir,
         const std::string &name = "",
         const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
