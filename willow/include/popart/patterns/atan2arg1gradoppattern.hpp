// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATAN2_ARG_1_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_ATAN2_ARG_1_GRAD_OP_PATTERN_HPP
#include <popart/patterns/binarygradoppattern.hpp>

namespace popart {

// Replace a Atan2ArgGradOp with -y/(x^2+y^2)
// {(fwd_in_y)} -> [Square] -> (tmp1)
// {(fwd_in_x)} -> [Square] -> (tmp2)
// {(tmp1), (tmp2)} -> [Add] -> (tmp3)
// {(fwd_in_y), (tmp3)} -> [Div] -> [ReduceSum] -> [Neg] -> (grad_out)
class Atan2Arg1GradOpPattern : public BinaryGradOpPattern {
public:
  bool matches(Op *) const override;

protected:
  virtual TensorId makeAllReplacementOps(Op *op,
                                         Ir *ir,
                                         const Tensor &gradIn,
                                         const Tensor &fwdIn0,
                                         const Tensor &fwdIn1,
                                         const Tensor &fwdOut) const override;
};

} // namespace popart

#endif
