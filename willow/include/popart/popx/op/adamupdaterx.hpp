// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAMUPDATERX_HPP
#define GUARD_NEURALNET_ADAMUPDATERX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class AdamUpdaterOpx : public PopOpx {
public:
  AdamUpdaterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // does not create inputs
};

} // namespace popx
} // namespace popart

#endif
