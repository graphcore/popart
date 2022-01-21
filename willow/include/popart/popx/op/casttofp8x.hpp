// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CASTTOFP8X_HPP
#define GUARD_NEURALNET_CASTTOFP8X_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class CastToFp8Opx : public PopOpx {
public:
  CastToFp8Opx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
