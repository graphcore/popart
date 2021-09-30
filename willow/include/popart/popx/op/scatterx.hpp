// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCATTERX_HPP
#define GUARD_NEURALNET_SCATTERX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

#include <popops/DynamicSlice.hpp>

namespace popart {
namespace popx {

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

private:
  int64_t axis;
};

class ScatterUpdateGradOpx : public PopOpx {
public:
  ScatterUpdateGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  int64_t axis;
};

} // namespace popx
} // namespace popart

#endif
