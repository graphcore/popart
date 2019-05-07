#ifndef GUARD_NEURALNET_SUBTRACT_HPP
#define GUARD_NEURALNET_SUBTRACT_HPP

#include <poponnx/op/elementwise.hpp>
#include <poponnx/op/reducesum.hpp>

namespace poponnx {

class SubtractOp : public ElementWiseBinaryOp {
public:
  SubtractOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class SubtractArg0GradOp : public ReduceSumOp {
public:
  SubtractArg0GradOp(const SubtractOp &, const std::vector<int64_t> &);
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
  SubtractArg1GradOp(const SubtractOp &);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  TensorInfo forward_op_arg_info;
};

} // namespace poponnx

#endif
