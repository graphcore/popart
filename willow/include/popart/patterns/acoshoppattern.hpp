// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACOSH_OP_PATTERN_HPP
#define GUARD_NEURALNET_ACOSH_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace AcoshOp
// acosh(x) = ln(x + sqrt(x^2 - 1) ); Defined for [1, +inf)

class AcoshOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
