#ifndef GUARD_NEURALNET_REDUCEMEAN_HPP
#define GUARD_NEURALNET_REDUCEMEAN_HPP

#include <popart/op.hpp>
#include <popart/op/reduce.hpp>

namespace popart {

class ReduceMeanOp : public ReduceOp {
public:
  ReduceMeanOp(const OperatorIdentifier &_opid,
               const std::vector<int64_t> &axes,
               const int64_t keepdims,
               const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReduceMeanGradOp : public ReduceGradOp {
public:
  ReduceMeanGradOp(const ReduceMeanOp &fwdOp, const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
