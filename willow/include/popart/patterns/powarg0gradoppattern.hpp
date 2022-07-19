// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_POWARG0GRADOPPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_POWARG0GRADOPPATTERN_HPP_
#include <popart/patterns/binarygradoppattern.hpp>

#include "popart/names.hpp"

namespace popart {
class Ir;
class Op;
class Tensor;

// Replace a PowArg0GradOp with
// {(fwd_in1), (ones)} -> [Minus] -> (tmp1)
// {(fwd_in0), (tmp1)} -> [Pow] -> (tmp2)
// {(fwd_in1), (tmp2)} -> [Mult] -> [ReduceSum] -> (grad_out)
class PowArg0GradOpPattern : public BinaryGradOpPattern {
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

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_POWARG0GRADOPPATTERN_HPP_
