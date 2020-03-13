// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_INITX_HPP
#define GUARD_NEURALNET_INITX_HPP

#include <popart/names.hpp>

namespace popart {
namespace popx {

class InitOpx : public Opx {
public:
  InitOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
