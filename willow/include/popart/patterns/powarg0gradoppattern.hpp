// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POW_ARG_0_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_POW_ARG_0_GRAD_OP_PATTERN_HPP

#include <popart/patterns/binarygradoppattern.hpp>

namespace popart {

// Replace a PowArg0GradOp with
// {(fwd_in1), (ones)} -> [Minus] -> (tmp1)
// {(fwd_in0), (tmp1)} -> [Pow] -> (tmp2)
// {(fwd_in1), (tmp2)} -> [Mult] -> [ReduceSum] -> (grad_out)
class PowArg0GradOpPattern : public BinaryGradOpPattern {
public:
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
