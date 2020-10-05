// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CUMSUMX_HPP
#define GUARD_NEURALNET_CUMSUMX_HPP

#include <popart/popx/opx.hpp>

#include <poplar/Tensor.hpp>

namespace popart {
namespace popx {

class CumSumOpx : public Opx {
public:
  CumSumOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  void CheckAxisValue() const;
  poplar::Tensor TriangularMatrix(std::size_t) const;
  int64_t ToNonNegativeAxis(int64_t) const;

  int64_t axis;
  int64_t exclusive;
  int64_t reverse;
};

} // namespace popx
} // namespace popart

#endif
