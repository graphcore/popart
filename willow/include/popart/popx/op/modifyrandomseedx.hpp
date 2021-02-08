
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MODIFYRANDOMSEEDX_HPP
#define GUARD_NEURALNET_MODIFYRANDOMSEEDX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class ModifyRandomSeedOp;

namespace popx {

class ModifyRandomSeedOpx : public Opx {
public:
  ModifyRandomSeedOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
