// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GATHERX_HPP
#define GUARD_NEURALNET_GATHERX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

#include <popops/DynamicSlice.hpp>

namespace popart {
namespace popx {

class GatherOpx : public Opx {
public:
  GatherOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // create the input poplar::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  poplar::Tensor createInput(int index,
                             const poplar::DebugNameAndId &dnai) const override;
  // default return DEADEND, i.e. unable to create input tensor, and
  // cannot use downstream opxs as candidates to create input
  // tensor
  InputCreatorType getInputCreatorType(int index0) const override;
  // If this Opx creates a poplar::Tensor at index0 (via createInput),
  // does it create the same poplar::Tensor as if opx1 creates one at
  // index1?. default behaviour : throws error
  bool createsEquiv(int index0, const Opx *opx1, int index1) const override;
  // To create a poplar::Tensor for input index index0, which
  // poplar::Tensors must already exist?
  std::set<TensorId> mustExistBeforeCreate(int index0) const override;

private:
  popops::SlicePlan plan;
  int64_t axis;
};

class GatherGradOpx : public Opx {
public:
  GatherGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  int64_t axis;
};

} // namespace popx
} // namespace popart

#endif
