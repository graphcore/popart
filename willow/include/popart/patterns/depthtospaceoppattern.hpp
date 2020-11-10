// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DEPTHTOSPACE_OP_PATTERN_HPP
#define GUARD_NEURALNET_DEPTHTOSPACE_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace DepthToSpaceOp with
// reshape and transpose.

class DepthToSpaceOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
