// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GATHERX_HPP
#define GUARD_NEURALNET_GATHERX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

#include <popops/DynamicSlice.hpp>

namespace popart {
namespace popx {

class GatherOpx : public PopOpx {
public:
  GatherOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // create the input snap::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const override;
  // default return DEADEND, i.e. unable to create input tensor, and
  // cannot use downstream opxs as candidates to create input
  // tensor
  InputCreatorType getInputCreatorType(int index0) const override;
  // To create a snap::Tensor for input index index0, which
  // snap::Tensors must already exist?
  std::set<TensorId> mustExistBeforeCreate(int index0) const override;

private:
  popops::SlicePlan plan;
  int64_t axis;
};

class GatherGradOpx : public PopOpx {
public:
  GatherGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  static std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
  handleNDMultiUpdate(poplar::Tensor target,
                      poplar::Tensor update,
                      poplar::Tensor indices,
                      int64_t axis);

private:
  int64_t axis;
};

} // namespace popx
} // namespace popart

#endif
