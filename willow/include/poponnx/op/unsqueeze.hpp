#ifndef GUARD_NEURALNET_UNSQUEEZE_HPP
#define GUARD_NEURALNET_UNSQUEEZE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class UnsqueezeOp : public Op {
public:
  UnsqueezeOp(const OperatorIdentifier &_opid,
              Ir *_ir,
              const std::string &name = "",
              const Attributes &_attr = {});
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  std::vector<int64_t> axes;
};

class UnsqueezeGradOp : public Op {
public:
  UnsqueezeGradOp(UnsqueezeOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  // The shape and type of the input to the constructing forward op
  TensorInfo squeezedInfo;
};

} // namespace poponnx

#endif
