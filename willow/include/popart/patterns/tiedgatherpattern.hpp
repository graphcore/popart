// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_TIEDGATHERPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_TIEDGATHERPATTERN_HPP_

#include <vector>
#include <popart/patterns/pattern.hpp>

#include "popart/names.hpp"

namespace popart {

class Op;
class Tensor;

class TiedGatherPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;

  std::vector<const Tensor *> touches(Op *) const override;

  bool apply(Op *) const override;
};

class TiedGatherAccumulatePattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;

  std::vector<const Tensor *> touches(Op *) const override;

  bool apply(Op *) const override;

private:
  TensorId inplaceTranspose(TensorId tid, Op *op) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_TIEDGATHERPATTERN_HPP_
