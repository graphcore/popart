// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ASINH_OP_PATTERN_HPP
#define GUARD_NEURALNET_ASINH_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace AsinhOp with
// asinh(x) = ln(x + sqrt(x^2 + 1))
// Notice: Absolute precision deteriorates for larger negative
// numbers as you will have ln(0.000001).

class AsinhOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
