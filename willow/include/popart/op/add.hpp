#ifndef GUARD_NEURALNET_ADD_HPP
#define GUARD_NEURALNET_ADD_HPP

#include <popart/op/elementwise.hpp>
#include <popart/op/reducesum.hpp>

namespace popart {

class AddOp : public ElementWiseBinaryOp {
public:
  AddOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;

  view::RegMap fwdRegMap(InIndex i) const override;
  view::RegMap bwdRegMap(InIndex i) const override;

  void setInplacePriority(const OperatorIdentifier &, float);

private:
  std::map<OperatorIdentifier, float> defaultInplacePriorities{
      {Onnx::CustomOperators::AddLhsInplace, 10.0f},
      {Onnx::CustomOperators::AddRhsInplace, 10.0f}};
};

class AddLhsInplaceOp : public AddOp {
public:
  AddLhsInplaceOp(const AddOp &addOp);

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {};
  }

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  view::Region modifies(InIndex index) const override;
  view::Region aliases(InIndex index) const override;
};

class AddRhsInplaceOp : public AddOp {
public:
  AddRhsInplaceOp(const AddOp &addOp);

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {};
  }

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  view::Region modifies(InIndex index) const override;
  view::Region aliases(InIndex index) const override;
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

} // namespace popart

#endif
