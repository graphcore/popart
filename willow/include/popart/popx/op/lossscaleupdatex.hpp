// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOSSSCALEUPDATEX_HPP
#define GUARD_NEURALNET_LOSSSCALEUPDATEX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class LossScaleUpdateOpx : public PopOpx {
public:
  LossScaleUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
