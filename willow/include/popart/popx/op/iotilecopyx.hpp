// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IOTILECOPYX_HPP
#define GUARD_NEURALNET_IOTILECOPYX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class IoTileCopyOpx : public PopOpx {
public:
  IoTileCopyOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  bool outputCreatedExternally(OutIndex) const final { return true; }
};

} // namespace popx
} // namespace popart

#endif
