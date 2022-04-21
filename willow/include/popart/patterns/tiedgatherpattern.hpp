// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TIED_GATHER_PATTERN_HPP
#define GUARD_NEURALNET_TIED_GATHER_PATTERN_HPP

#include <popart/patterns/pattern.hpp>

#include <map>

namespace popart {

class Op;
class MatMulBaseOp;

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

#endif