// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_TILEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_TILEX_HPP_

#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class TileOpx : public Opx {
public:
  TileOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // Design decision: InputCreatorType could be CANUNWIND, but not overriding
  // default, DEADEND. The unwind function would slice the output over each
  // dimension with a non-idendity repeat value. This could result in allocating
  // a much larger tensor than required by the input's shape
};

class TileGradOpx : public Opx {
public:
  TileGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_TILEX_HPP_
