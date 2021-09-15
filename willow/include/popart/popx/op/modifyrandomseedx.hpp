
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MODIFYRANDOMSEEDX_HPP
#define GUARD_NEURALNET_MODIFYRANDOMSEEDX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

class ModifyRandomSeedOp;

namespace popx {

class ModifyRandomSeedOpx : public PopOpx {
public:
  ModifyRandomSeedOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
