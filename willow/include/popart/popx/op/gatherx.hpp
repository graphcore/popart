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

  // create the input poplar::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  // default return DEADEND, i.e. unable to create input tensor, and
  // cannot use downstream opxs as candidates to create input
  // tensor
  InputCreatorType getInputCreatorType(InIndex index) const final;
  // To create a poplar::Tensor for the given input index, which
  // poplar::Tensors must already exist?
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  popops::SlicePlan plan;
  int64_t axis;
};

class GatherGradOpx : public PopOpx {
public:
  GatherGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // create the input poplar::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  // default return DEADEND, i.e. unable to create input tensor, and
  // cannot use downstream opxs as candidates to create input
  // tensor
  InputCreatorType getInputCreatorType(InIndex index) const final;
  // To create a poplar::Tensor for the given input index, which
  // poplar::Tensors must already exist?
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  popops::SlicePlan plan;
  int64_t axis;
};

} // namespace popx
} // namespace popart

#endif
