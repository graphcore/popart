// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GROUPNORMX_HPP
#define GUARD_NEURALNET_GROUPNORMX_HPP

#include <popart/popx/op/normx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class GroupNormOpx : public NormOpx {
public:
  GroupNormOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
};

class GroupNormGradOpx : public NormOpx {
public:
  GroupNormGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
};

} // namespace popx
} // namespace popart

#endif
