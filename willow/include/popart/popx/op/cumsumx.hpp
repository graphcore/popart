// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CUMSUMX_HPP
#define GUARD_NEURALNET_CUMSUMX_HPP

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

class CumSumOpx : public PopOpx {
public:
  CumSumOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class CumSumGradOpx : public PopOpx {
public:
  CumSumGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
