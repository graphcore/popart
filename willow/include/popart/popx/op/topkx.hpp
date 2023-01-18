// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_TOPKX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_TOPKX_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <popart/popx/op/basesortx.hpp>

#include "popart/popx/opx.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {

namespace popops {
class SlicePlan;
} // namespace popops

namespace program {
class Sequence;
} // namespace program

} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class TopKOpx : public BaseSortOpx {
public:
  TopKOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  unsigned K;
};

class TopKGradOpx : public Opx {
public:
  TopKGradOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }

private:
  int64_t axis;
  TensorInfo gradOutInfo;

  popops::SlicePlan plan;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_TOPKX_HPP_
