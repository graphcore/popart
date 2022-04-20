
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MODIFYRANDOMSEEDX_HPP
#define GUARD_NEURALNET_MODIFYRANDOMSEEDX_HPP

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

#endif
