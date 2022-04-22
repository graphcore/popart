// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMUNIFORMX_HPP
#define GUARD_NEURALNET_RANDOMUNIFORMX_HPP

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

class RandomUniformOpx : public PopOpx {
public:
  RandomUniformOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
