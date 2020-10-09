// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CUMSUMX_HPP
#define GUARD_NEURALNET_CUMSUMX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class CumSumOpx : public Opx {
public:
  CumSumOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class CumSumGradOpx : public Opx {
public:
  CumSumGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
