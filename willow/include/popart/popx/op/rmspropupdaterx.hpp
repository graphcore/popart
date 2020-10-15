// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RMSPROPUPDATERX_HPP
#define GUARD_NEURALNET_RMSPROPUPDATERX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class RMSPropUpdaterOpx : public Opx {
public:
  RMSPropUpdaterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // does not create inputs
};

} // namespace popx
} // namespace popart

#endif
