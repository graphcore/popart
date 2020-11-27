// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POW_ARG_1_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_POW_ARG_1_GRAD_OP_PATTERN_HPP
#include <popart/patterns/binarygradoppattern.hpp>

namespace popart {

// Replace a PowArg1GradOp with
// (fwd_in0) -> [Log] -> (tmp1)
// {(out), (tmp1)} -> [Mult] -> [ReduceSum] -> (grad_out)
class PowArg1GradOpPattern : public BinaryGradOpPattern {
public:
  // Does op at the root of the
  // pattern make a match?
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
