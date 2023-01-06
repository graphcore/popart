// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDADDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDADDX_HPP_

#include <poplar/Tensor.hpp>
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

class ScaledAddOpx : public Opx {
public:
  ScaledAddOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

protected:
  poplar::Tensor compute(poplar::program::Sequence &prog,
                         poplar::Tensor in0,
                         poplar::Tensor in1,
                         poplar::Tensor s0,
                         poplar::Tensor s1,
                         float s0f,
                         float s1f,
                         bool inplace) const;
};

class ScaledAddLhsInplaceOpx : public ScaledAddOpx {
public:
  ScaledAddLhsInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ScaledAddRhsInplaceOpx : public ScaledAddOpx {
public:
  ScaledAddRhsInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDADDX_HPP_
