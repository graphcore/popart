// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SYNCX_HPP
#define GUARD_NEURALNET_SYNCX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {
class SyncOpx : public PopOpx {
public:
  SyncOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
