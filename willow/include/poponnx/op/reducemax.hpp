#ifndef GUARD_NEURALNET_REDUCEMAX_HPP
#define GUARD_NEURALNET_REDUCEMAX_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/reduce.hpp>

namespace poponnx {

class ReduceMaxOp : public ReduceOp {
public:
  ReduceMaxOp(const OperatorIdentifier &_opid,
              const std::vector<int64_t> &axes,
              const int64_t keepdims,
              const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReduceMaxGradOp : public ReduceGradOp {
public:
  ReduceMaxGradOp(const ReduceMaxOp &fwdOp, const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  static InIndex getFwdInInIndex() { return 1; }
  static InIndex getFwdOutInIndex() { return 2; }
};

} // namespace poponnx

#endif
