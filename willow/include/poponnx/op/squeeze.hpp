#ifndef GUARD_NEURALNET_SQUEEZE_HPP
#define GUARD_NEURALNET_SQUEEZE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class SqueezeOp : public Op {
public:
  SqueezeOp(const OperatorIdentifier &_opid,
            Ir *_ir,
            const std::string &name = "",
            const Attributes &_attr = {});
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

class SqueezeGradOp : public Op {
public:
  SqueezeGradOp(SqueezeOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  // The shape and type of the input to the constructing forward op
  TensorInfo unsqueezedInfo;
};

} // namespace poponnx

#endif
