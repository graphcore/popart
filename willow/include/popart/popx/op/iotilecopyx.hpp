// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IOTILECOPYX_HPP
#define GUARD_NEURALNET_IOTILECOPYX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class IoTileCopyOpx : public Opx {
public:
  IoTileCopyOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
