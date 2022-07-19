// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_EXPANDCAST_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_EXPANDCAST_HPP_

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

// Replaces an expand followed by a cast with a more efficient cast followed by
// expand. As the cast always copies, the expand can generally always be
// inplace. The pattern does not apply where this is not the case e.g. another
// consumer or different IPUs.

class ExpandCastPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_EXPANDCAST_HPP_
