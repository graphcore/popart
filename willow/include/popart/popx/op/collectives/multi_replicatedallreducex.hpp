// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCEX_HPP_
#include <poplar/Tensor.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

#include "popart/names.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

/**
 * Lowers the MultiReplicatedAllReduceOp to Poplar by growing each
 * individual output tensor, and performing a to-destination all-reduce
 * on a concatenation of the input tensors. Mixing of both in-place and
 * out-place all-reduce operations is supported.
 */
class MultiReplicatedAllReduceOpx : public MultiCollectiveBaseOpx {
public:
  MultiReplicatedAllReduceOpx(popart::Op *op, Devicex *devicex);
  InputCreatorType getInputCreatorType(InIndex) const override;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex in,
                                    OutIndex out) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
  void growPart(OpxGrowPartId id) const override;
  void grow(poplar::program::Sequence &prog) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCEX_HPP_
