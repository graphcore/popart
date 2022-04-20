// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEMINX_HPP
#define GUARD_NEURALNET_REDUCEMINX_HPP

#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class ReduceMinOpx : public PopOpx {
public:
  ReduceMinOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceMinGradOpx : public PopOpx {
public:
  ReduceMinGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
