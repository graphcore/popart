// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTX_HPP_

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

class CastOpx : public Opx {
public:
  CastOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class CastGradOpx : public CastOpx {
public:
  CastGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTX_HPP_
