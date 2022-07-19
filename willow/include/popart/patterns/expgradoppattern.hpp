// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_EXPGRADOPPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_EXPGRADOPPATTERN_HPP_

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

// Replace a ExpGradOp with
// grad_out = grad_in * fwd_out
class ExpGradOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_EXPGRADOPPATTERN_HPP_
