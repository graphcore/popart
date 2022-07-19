// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDADDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDADDX_HPP_

#include <snap/Tensor.hpp>
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

class ScaledAddOpx : public PopOpx {
public:
  ScaledAddOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;

protected:
  snap::Tensor compute(snap::program::Sequence &prog,
                       snap::Tensor in0,
                       snap::Tensor in1,
                       snap::Tensor s0,
                       snap::Tensor s1,
                       float s0f,
                       float s1f,
                       bool inplace) const;
};

class ScaledAddLhsInplaceOpx : public ScaledAddOpx {
public:
  ScaledAddLhsInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class ScaledAddRhsInplaceOpx : public ScaledAddOpx {
public:
  ScaledAddRhsInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDADDX_HPP_
