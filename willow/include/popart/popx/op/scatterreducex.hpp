// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERREDUCEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERREDUCEX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <cstddef>
#include <set>
#include <snap/Tensor.hpp>
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
class ReductionStrategy;

class ScatterReduceOpx : public PopOpx {
public:
  ScatterReduceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  std::unique_ptr<ReductionStrategy> strategy;
  popops::SlicePlan plan;
  size_t axis;
};

class ScatterReduceGradOpx : public PopOpx {
public:
  ScatterReduceGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  std::unique_ptr<ReductionStrategy> strategy;
  popops::SlicePlan plan;
  size_t axis;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERREDUCEX_HPP_
