
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MODIFYRANDOMSEEDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MODIFYRANDOMSEEDX_HPP_

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

class ModifyRandomSeedOpx : public PopOpx {
public:
  ModifyRandomSeedOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MODIFYRANDOMSEEDX_HPP_
