// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESHAPE_COLLAPSE_PATTERN_HPP
#define GUARD_NEURALNET_RESHAPE_COLLAPSE_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Modify inputs of reshapes from:
//  y = reshape(x)
//  z = reshape(y)
// to:
//  y = reshape(x)
//  z = reshape(x)

class ReshapeCollapsePattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif