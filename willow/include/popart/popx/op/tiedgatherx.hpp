// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TIEDGATHERX_HPP
#define GUARD_NEURALNET_TIEDGATHERX_HPP

#include <popart/popx/op/gatherx.hpp>

namespace popart {
namespace popx {

class TiedGatherOpx final : public GatherBaseOpx {
public:
  TiedGatherOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(int index0) const final;

  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
};

} // namespace popx
} // namespace popart

#endif
