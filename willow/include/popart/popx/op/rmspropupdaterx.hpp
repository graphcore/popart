// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_RMSPROPUPDATERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_RMSPROPUPDATERX_HPP_

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

class RMSPropUpdaterOpx : public PopOpx {
public:
  RMSPropUpdaterOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // does not create inputs
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_RMSPROPUPDATERX_HPP_
