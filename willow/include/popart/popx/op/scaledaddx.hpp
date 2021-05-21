// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCALEDADDX_HPP
#define GUARD_NEURALNET_SCALEDADDX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class ScaledAddOpx : public PopOpx {
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

#endif
