// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ABORTX_HPP
#define GUARD_NEURALNET_ABORTX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {
class AbortOpx : public Opx {
public:
  AbortOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
