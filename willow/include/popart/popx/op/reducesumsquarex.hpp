// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_REDUCESUMSQUAREX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_REDUCESUMSQUAREX_HPP_

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

class ReduceSumSquareOpx : public Opx {
public:
  ReduceSumSquareOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceSumSquareGradOpx : public Opx {
public:
  ReduceSumSquareGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_REDUCESUMSQUAREX_HPP_
