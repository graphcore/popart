// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NOPX_HPP
#define GUARD_NEURALNET_NOPX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class NopOpx : public Opx {
public:
  NopOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
