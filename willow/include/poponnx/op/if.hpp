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
       const std::vector<TensorId> &then_output_ids,
       const std::vector<TensorId> &else_output_ids,
       const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getConditionInIndex() { return 0; }

  Scope getThenScope() const;
  Scope getElseScope() const;
  const Graph &getThenGraph() const;
  const Graph &getElseGraph() const;

  const std::vector<TensorId> &getThenInputIds() { return then_input_ids; }
  const std::vector<TensorId> &getElseInputIds() { return else_input_ids; }
  const std::vector<TensorId> &getThenOutputIds() { return then_output_ids; }
  const std::vector<TensorId> &getElseOutputIds() { return else_output_ids; }

  bool isOutlineable() const override { return false; }

private:
  void appendInputs(const std::vector<TensorId> &input_ids, const Scope &);

  const std::vector<TensorId> then_input_ids;
  const std::vector<TensorId> else_input_ids;
  const std::vector<TensorId> then_output_ids;
  const std::vector<TensorId> else_output_ids;
};

} // namespace poponnx

#endif
