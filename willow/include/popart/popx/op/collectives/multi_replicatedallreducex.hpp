// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIREPLICATEDALLREDUCEX_HPP
#define GUARD_NEURALNET_MULTIREPLICATEDALLREDUCEX_HPP
#include <popart/op/collectives/multi_replicatedallreduce.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

namespace popart {
namespace popx {

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

#endif