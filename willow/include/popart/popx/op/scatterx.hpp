// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERX_HPP_

#include <cstdint>
#include <set>
#include <poplar/Tensor.hpp>
#include <popops/DynamicSlice.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class ScatterOpx : public Opx {
public:
  ScatterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  popops::SlicePlan plan;
  int64_t axis;
};

class ScatterDataGradOpx : public Opx {
public:
  ScatterDataGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  popops::SlicePlan plan;
  int64_t axis;
};

class ScatterUpdateGradOpx : public Opx {
public:
  ScatterUpdateGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  popops::SlicePlan plan;
  int64_t axis;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERX_HPP_
