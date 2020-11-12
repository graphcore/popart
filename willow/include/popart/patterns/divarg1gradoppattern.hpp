// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DIV_ARG_1_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_DIV_ARG_1_GRAD_OP_PATTERN_HPP

#include <popart/patterns/binarygradoppattern.hpp>

namespace popart {

// Replace a DivArg1GradOp with
// (fwd_in1) -> [Square] -> (tmp1)
// {(grad_in), (fwd_in0)} -> [Mul] -> (tmp2)
// {(tmp2), (tmp1)} -> [Div] -> [Negate] -> [ReduceSum] -> (grad_out)
class DivArg1GradOpPattern : public BinaryGradOpPattern {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(Op *) const override;

protected:
  virtual TensorId makeAllReplacementOps(Op *op,
                                         Tensor *grad_in,
                                         Tensor *fwd_in0,
                                         Tensor *fwd_in1,
                                         Tensor *fwd_out) const override;
};

} // namespace popart

#endif
