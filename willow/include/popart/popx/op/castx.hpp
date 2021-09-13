// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CASTX_HPP
#define GUARD_NEURALNET_CASTX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class CastOpx : public PopOpx {
public:
  CastOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class CastGradOpx : public CastOpx {
public:
  CastGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
