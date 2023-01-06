// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1NESTEROVX_HPP
#define GUARD_NEURALNET_SGD1NESTEROVX_HPP

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

class SGD1NesterovOpx : public Opx {
public:
  SGD1NesterovOpx(Op *, Devicex *);
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

} // namespace popx
} // namespace popart

#endif
