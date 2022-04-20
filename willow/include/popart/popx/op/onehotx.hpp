// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONEHOTX_HPP
#define GUARD_NEURALNET_ONEHOTX_HPP

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

class OnehotOpx : public PopOpx {
public:
  OnehotOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class OnehotGradOpx : public PopOpx {
public:
  OnehotGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
