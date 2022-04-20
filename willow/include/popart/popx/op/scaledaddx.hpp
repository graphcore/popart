// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCALEDADDX_HPP
#define GUARD_NEURALNET_SCALEDADDX_HPP

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

#endif
