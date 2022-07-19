// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_OPTOIDENTITYPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_OPTOIDENTITYPATTERN_HPP_

#include <memory>
#include <vector>
#include <popart/patterns/sequenceexpander.hpp>

namespace popart {
class Op;

// Replace ops that return their only input unchanged with an identity op
class OpToIdentityPattern : public SequenceExpander {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(Op *) const override;
  // what phase should this Pattern run in? PRETOPOCONS, as it does not
  // handle topological constraints.

private:
  // Replace the given op with the returned sequence of ops
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_OPTOIDENTITYPATTERN_HPP_
