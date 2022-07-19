// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTX_HPP_

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

class CastOpx : public PopOpx {
public:
  CastOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class CastGradOpx : public CastOpx {
public:
  CastGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTX_HPP_
