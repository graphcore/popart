// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_TRANSPOSETOIDENTITYORRESHAPE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_TRANSPOSETOIDENTITYORRESHAPE_HPP_

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

// Replaces:
//  Transpose -> {Identity/Reshape} -> Transpose
//          with
//  Identity/Reshape

class TransposeToIdentityOrReshapePattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_TRANSPOSETOIDENTITYORRESHAPE_HPP_
