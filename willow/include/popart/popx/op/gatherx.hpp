// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_GATHERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_GATHERX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <cstdint>
#include <set>
#include <snap/Tensor.hpp>
#include <tuple>
#include <poplar/Tensor.hpp>
#include <popops/DynamicSlice.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class GatherBaseOpx : public PopOpx {
public:
  GatherBaseOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override = 0;

  // create the input snap::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const override = 0;

  // default return DEADEND, i.e. unable to create input tensor, and
  // cannot use downstream opxs as candidates to create input
  // tensor
  InputCreatorType getInputCreatorType(int index0) const override = 0;

  // To create a poplar::Tensor for input index index0, which
  // poplar::Tensors must already exist?
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;

protected:
  int64_t axis;
  void setCommonMembersPostVerify(const Op *op);

  // Zero indices that are out of range so they produce output from the weight
  // tensor
  std::tuple<poplar::Tensor, poplar::Tensor>
  zeroIndiciesThatAreOutOfRange(snap::program::Sequence &prog,
                                const poplar::Tensor &data,
                                const poplar::Tensor &offsets) const;

  // Zero output corresponding to out of range indices
  void zeroOutputOfOutOfRangeIndices(snap::program::Sequence &prog,
                                     poplar::Tensor &result,
                                     const poplar::Tensor &mask,
                                     const poplar::Tensor &data) const;
};

class GatherOpx final : public GatherBaseOpx {
public:
  GatherOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // create the input poplar::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  snap::Tensor
  createInputTensor(int index, const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(int index) const final;

private:
  popops::SlicePlan plan;
};

class GatherGradOpx : public PopOpx {
public:
  GatherGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  static std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
  handleNDMultiUpdate(poplar::Tensor target,
                      poplar::Tensor update,
                      poplar::Tensor indices,
                      int64_t axis);
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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_GATHERX_HPP_
