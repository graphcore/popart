// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOG1P_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_LOG1P_GRAD_OP_PATTERN_HPP

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

// Replace a Log1pGradOp with
// grad_out = grad_in / (fwd_in + 1)
class Log1pGradOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
