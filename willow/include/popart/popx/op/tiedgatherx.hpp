// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TIEDGATHERX_HPP
#define GUARD_NEURALNET_TIEDGATHERX_HPP

#include "popart/popx/debugcontextx.hpp"
#include <snap/Tensor.hpp>
#include <popart/popx/op/gatherx.hpp>

#include "popart/names.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

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
