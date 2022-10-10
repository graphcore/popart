// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_SLICEOPPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_SLICEOPPATTERN_HPP_

#include <vector>
#include <popart/patterns/pattern.hpp>

// support slice op when 'step > 1 or step < -1'
// example: slice input [0,1,2,3,4], axe = 0, [start,end)=[4,0), step=-2
// step1: Slice, step=-1 ,slice result=[4,3,2,1];
// step2: Subsample, stride=2, final result is [4,2]

namespace popart {
class Op;
class Tensor;

class SlicePattern : public PreAliasPattern {
public:
  bool matches(Op *op) const override;

  std::vector<const Tensor *> touches(Op *) const override { return {}; }

  bool apply(Op *op) const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_SLICEOPPATTERN_HPP_
