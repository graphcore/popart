// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_FMOD_ARG_0_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_FMOD_ARG_0_GRAD_OP_PATTERN_HPP

#include <popart/patterns/binarygradoppattern.hpp>

#include "popart/names.hpp"

namespace popart {
class Ir;
class Op;
class Tensor;

// Replace a FmodArg0GradOp with a constant tensor filled with ones that has the
// same shape and type as its input. This matches the behavior of PyTorch.
class FmodArg0GradOpPattern : public BinaryGradOpPattern {
public:
  bool matches(Op *) const final;

protected:
  TensorId makeAllReplacementOps(Op *op,
                                 Ir *ir,
                                 const Tensor &gradIn,
                                 const Tensor &fwdIn0,
                                 const Tensor &fwdIn1,
                                 const Tensor &fwdOut) const final;
};

} // namespace popart

#endif // !GUARD_NEURALNET_FMOD_ARG_0_GRAD_OP_PATTERN_HPP
