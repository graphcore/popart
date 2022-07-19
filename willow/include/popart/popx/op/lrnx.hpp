// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LRNX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LRNX_HPP_

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

class LRNOpx : public PopOpx {
public:
  LRNOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class LRNGradOpx : public PopOpx {
public:
  LRNGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LRNX_HPP_
