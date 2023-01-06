// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_REDUCEL2X_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_REDUCEL2X_HPP_

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

class ReduceL2Opx : public Opx {
public:
  ReduceL2Opx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceL2GradOpx : public Opx {
public:
  ReduceL2GradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_REDUCEL2X_HPP_
