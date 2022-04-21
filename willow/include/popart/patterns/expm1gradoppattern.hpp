// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXPM1_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_EXPM1_GRAD_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace a Expm1GradOp with
// grad_out = grad_in * (fwd_out + 1) = grad_in * exp(x)
class Expm1GradOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
