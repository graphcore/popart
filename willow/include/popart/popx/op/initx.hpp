// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_INITX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_INITX_HPP_

#include <popart/names.hpp>
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

class InitOpx : public PopOpx {
public:
  InitOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  bool outputCreatedExternally(OutIndex) const final { return true; }
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_INITX_HPP_
