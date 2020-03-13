// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SPLIT_GATHER_PATTERN_HPP
#define GUARD_NEURALNET_SPLIT_GATHER_PATTERN_HPP

#include <popart/patterns/pattern.hpp>

namespace popart {

class SplitGatherPattern : public PreAliasPattern {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(Op *) const override;
  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  std::vector<const Tensor *> touches(Op *) const override;
  // apply the pattern,
  // changes the graph of the op
  bool apply(Op *) const override;
};

} // namespace popart

#endif
