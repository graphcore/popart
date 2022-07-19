// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <cstdint>
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

class ScatterOpx : public PopOpx {
public:
  ScatterOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  popops::SlicePlan plan;
  int64_t axis;
};

class ScatterDataGradOpx : public PopOpx {
public:
  ScatterDataGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  popops::SlicePlan plan;
  int64_t axis;
};

class ScatterUpdateGradOpx : public PopOpx {
public:
  ScatterUpdateGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  snap::Tensor
  createInputTensor(InIndex index,
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
