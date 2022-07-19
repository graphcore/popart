// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ROIALIGNX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ROIALIGNX_HPP_

#include <poplar/Tensor.hpp>

#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class RoiAlignOpx : public Opx {
public:
  RoiAlignOpx(Op *, Devicex *);
  ~RoiAlignOpx() override = default;
  virtual void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor roiAlignImpl(poplar::program::Sequence &prog,
                              poplar::Tensor &bottomData,
                              poplar::Tensor &bottomRois,
                              poplar::Tensor &bottomBatchIndex) const;
};

class RoiAlignGradOpx : public Opx {
public:
  RoiAlignGradOpx(Op *, Devicex *);
  ~RoiAlignGradOpx() override = default;
  virtual void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor roiAlignImpl(poplar::program::Sequence &prog,
                              poplar::Tensor &topDiff,
                              poplar::Tensor &bottomData,
                              poplar::Tensor &bottomRois,
                              poplar::Tensor &bottomBatchIndex) const;
};

} // namespace popx
} // namespace popart
#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ROIALIGNX_HPP_
