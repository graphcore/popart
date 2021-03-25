// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCATTERREDUCEX_HPP
#define GUARD_NEURALNET_SCATTERREDUCEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

#include <popops/DynamicSlice.hpp>

namespace popart {
namespace popx {

class ScatterReduceOpx : public Opx {
public:
  ScatterReduceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;

  bool createsEquiv(int index0, const Opx *opx1, int index1) const final {
    return false;
  }

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  popops::SlicePlan plan;
  size_t axis;
};

class ScatterReduceGradOpx : public Opx {
public:
  ScatterReduceGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  int64_t axis;
};

} // namespace popx
} // namespace popart

#endif
