// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERREDUCEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERREDUCEX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <cstddef>
#include <set>
#include <popops/DynamicSlice.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;
class ReductionStrategy;

class ScatterReduceOpx : public Opx {
public:
  ScatterReduceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  std::unique_ptr<ReductionStrategy> strategy;
  popops::SlicePlan plan;
  size_t axis;
  size_t group_size;
};

class ScatterReduceGradOpx : public Opx {
public:
  ScatterReduceGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  std::unique_ptr<ReductionStrategy> strategy;
  popops::SlicePlan plan;
  size_t axis;
  size_t group_size;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERREDUCEX_HPP_
