#ifndef GUARD_NEURALNET_IF_HPP
#define GUARD_NEURALNET_IF_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class IfOp : public Op {
public:
  IfOp(const OperatorIdentifier &,
       const std::vector<TensorId> &then_input_ids,
       const std::vector<TensorId> &else_input_ids,
       const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  // The number of outputs of the else and then branches
  // The number of inputs to the IfOp is:
  //   2 * inputsPerBranch + 1 (the condition input)
  int inputsPerBranch() const;
  void setup() final;

  static InIndex getConditionInIndex() { return 0; }
  InIndex getThenBranchInIndex(InIndex i) const { return i + 1; }
  InIndex getElseBranchInIndex(InIndex i) const {
    return i + inputsPerBranch() + 1;
  }

  Scope getThenScope();
  Scope getElseScope();

private:
  void appendInputs(const std::vector<TensorId> &input_ids, const Scope &);

  const std::vector<TensorId> then_input_ids;
  const std::vector<TensorId> else_input_ids;
};

} // namespace poponnx

#endif
