// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCEX_HPP_
#include <snap/Tensor.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

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

class MultiReplicatedAllReduceOpx : public MultiCollectiveBaseOpx {
public:
  MultiReplicatedAllReduceOpx(popart::Op *op, Devicex *devicex);
  InputCreatorType getInputCreatorType(InIndex) const override;
  snap::Tensor unwindTensorLayout(snap::Tensor tensor,
                                  InIndex in,
                                  OutIndex out) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
  void growPart(OpxGrowPartId id) const override;
  void grow(snap::program::Sequence &prog) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCEX_HPP_
