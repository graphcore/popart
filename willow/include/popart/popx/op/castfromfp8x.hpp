// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CASTFROMFP8X_HPP
#define GUARD_NEURALNET_CASTFROMFP8X_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class CastFromFp8Opx : public PopOpx {
public:
  CastFromFp8Opx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
