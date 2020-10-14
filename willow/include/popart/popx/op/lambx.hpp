// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LAMBX_HPP
#define GUARD_NEURALNET_LAMBX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class LambSquareOpx : public Opx {
public:
  LambSquareOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
