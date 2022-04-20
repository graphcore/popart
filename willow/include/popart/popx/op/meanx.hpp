// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MEANX_HPP
#define GUARD_NEURALNET_MEANX_HPP

#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class MeanOpx : public ElementWiseUnaryOpx {
public:
  MeanOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class MeanArgGradOpx : public PopOpx {
public:
  MeanArgGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
