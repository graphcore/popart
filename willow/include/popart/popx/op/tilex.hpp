// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TILEX_HPP
#define GUARD_NEURALNET_TILEX_HPP

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

class TileOpx : public PopOpx {
public:
  TileOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // Design decision: InputCreatorType could be CANUNWIND, but not overriding
  // default, DEADEND. The unwind function would slice the output over each
  // dimension with a non-idendity repeat value. This could result in allocating
  // a much larger tensor than required by the input's shape
};

class TileGradOpx : public PopOpx {
public:
  TileGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
