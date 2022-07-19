// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MEANX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MEANX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MEANX_HPP_
