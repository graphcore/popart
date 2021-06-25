// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PACKEDDATABLOCK_PATTERN_HPP
#define GUARD_NEURALNET_PACKEDDATABLOCK_PATTERN_HPP

#include <popart/patterns/pattern.hpp>

namespace popart {

class PackedDataBlockPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
