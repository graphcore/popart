// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBTRACT_ARG1_GRAD_PATTERN_HPP
#define GUARD_NEURALNET_SUBTRACT_ARG1_GRAD_PATTERN_HPP
#include <popart/patterns/binarygradoppattern.hpp>

namespace popart {

// Replace a SubtractArg1GradOp with a negate followed by a reducesum
class SubtractArg1GradOpPattern : public BinaryGradOpPattern {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(Op *) const override;
  // what phase should this Pattern run in? PRETOPOCONS, as it does not
  // handle topological constraints.

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
