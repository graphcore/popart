#ifndef GUARD_NEURALNET_SUBTRACT_HPP
#define GUARD_NEURALNET_SUBTRACT_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/reducesum.hpp>

namespace poponnx {

class SubtractOp : public Op {
public:
  SubtractOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

class SubtractArg0GradOp : public ReduceSumOp {
public:
  SubtractArg0GradOp(SubtractOp *, const std::vector<int64_t> &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  TensorInfo forward_op_arg_info;
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class SubtractArg1GradOp : public Op {
public:
  SubtractArg1GradOp(SubtractOp *);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  TensorInfo forward_op_arg_info;
};

} // namespace poponnx

#endif
