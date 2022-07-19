// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_VIEWSIMPLIFYPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_VIEWSIMPLIFYPATTERN_HPP_

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

// Simplify "chains" of view changing ops from:
//  y = a(x)
//  z = b(y)
// to:
//  y = a(x)
//  z = c(x)
// where:
//   if a is reshape and b is identity:
//     c = reshape
//   else:
//     c = b
//
// Only supports IdentityOp and ReshapeBaseOp.
// TODO(T35507): Generalize to chains of any view changing operation

class ViewSimplifyPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_VIEWSIMPLIFYPATTERN_HPP_
