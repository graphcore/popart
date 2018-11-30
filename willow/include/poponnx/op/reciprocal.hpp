#ifndef GUARD_NEURALNET_RECIPROCAL_HPP
#define GUARD_NEURALNET_RECIPROCAL_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class ReciprocalOp : public Op {
public:
  ReciprocalOp(const OpConstructorBundle &);
  ReciprocalOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
};

class ReciprocalGradOp : public Op {
public:
  ReciprocalGradOp(ReciprocalOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
};

} // namespace poponnx

#endif
