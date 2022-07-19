// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_SPLITGRADOPTOCONCATPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_SPLITGRADOPTOCONCATPATTERN_HPP_

#include <memory>
#include <vector>
#include <popart/patterns/sequenceexpander.hpp>

namespace popart {
class Op;

// Replace ops that return their only input unchanged with an identity op
class SplitGradOpToConcatPattern : public SequenceExpander {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(Op *) const override;

private:
  // Replace the given op with the returned sequence of ops
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_SPLITGRADOPTOCONCATPATTERN_HPP_
