#ifndef GUARD_NEURALNET_ADD_HPP
#define GUARD_NEURALNET_ADD_HPP

#include <poponnx/op/elementwise.hpp>
#include <poponnx/op/reducesum.hpp>

namespace poponnx {

class AddOp : public ElementWiseBinaryOp {
public:
  AddOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class AddArg0GradOp : public ReduceSumOp {
public:
  AddArg0GradOp(const AddOp &, const std::vector<int64_t> &axes);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  TensorInfo forward_op_arg_info;
};

class AddArg1GradOp : public ReduceSumOp {
public:
  AddArg1GradOp(const AddOp &, const std::vector<int64_t> &axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  TensorInfo forward_op_arg_info;
};

} // namespace poponnx

#endif
