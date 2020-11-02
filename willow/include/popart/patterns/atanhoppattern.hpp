// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATANH_OP_PATTERN_HPP
#define GUARD_NEURALNET_ATANH_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace AtanhOp
// atanh(x) = 1/2 ln( (1 + x) / (1 - x) )

class AtanhOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
