// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ABORTX_HPP
#define GUARD_NEURALNET_ABORTX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {
class AbortOpx : public PopOpx {
public:
  AbortOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
