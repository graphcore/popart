#ifndef GUARD_NEURALNET_ADD_HPP
#define GUARD_NEURALNET_ADD_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/reducesum.hpp>

namespace poponnx {

class AddOp : public Op {
public:
  AddOp(const OperatorIdentifier &_opid,
        Ir *_ir,
        const std::string &name = "",
        const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

class AddArg0GradOp : public ReduceSumOp {
public:
  AddArg0GradOp(AddOp *, const std::vector<int64_t> &axes);
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
  AddArg1GradOp(AddOp *, const std::vector<int64_t> &axes);
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
