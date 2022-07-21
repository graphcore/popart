// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIREPLICATEDREDUCESCATTERX_HPP
#define GUARD_NEURALNET_MULTIREPLICATEDREDUCESCATTERX_HPP
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
/**
 * Lowers the MultiReplicatedReduceScatterOp to Poplar by growing each
 * individual output tensor, and performing a to-destination reduce-scatter
 * on a concatenation of the input tensors. CBR features and logic
 * are transplanted from the ReplicatedReduceScatterOpx implementation
 */
class MultiReplicatedReduceScatterOpx : public MultiCollectiveBaseOpx {
public:
  MultiReplicatedReduceScatterOpx(popart::Op *op, Devicex *devicex);
  InputCreatorType getInputCreatorType(InIndex) const override;
  void growPart(OpxGrowPartId id) const override;
  void grow(snap::program::Sequence &prog) const override;
  ViewChangers getCreatorViewChangers(InIndex index) const override;
  DnfTensorIds mustExistBeforeCreateDNF(InIndex in) const override;
  snap::Tensor prepareInputForGCL(snap::Tensor inputTensor,
                                  InIndex inputIndex,
                                  snap::program::Sequence &prog) const;
  snap::Tensor
  createInputTensor(InIndex idx,
                    const poplar::DebugNameAndId &dnai) const override;
};

} // namespace popx
} // namespace popart

#endif
