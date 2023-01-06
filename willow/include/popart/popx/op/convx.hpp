// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVX_HPP_

#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/convbasex.hpp>
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

class ConvOpx : public MultiConvBaseOpx {
public:
  ConvOpx(Op *, Devicex *);
  poplar::Tensor createWeightsInput(const poplar::DebugNameAndId &dnai,
                                    int convIndex) const final;
  poplar::Tensor createDataInput(const poplar::DebugNameAndId &dnai,
                                 int convIndex) const final;
  InputCreatorType getInputCreatorType(InIndex idx) const override;
  std::vector<poplar::Tensor>
  convolve(poplar::program::Sequence &,
           const std::vector<poplar::Tensor> &weights) const final;
};

class ConvWeightsGradOpx : public MultiConvWeightsGradBaseOpx {
public:
  ConvWeightsGradOpx(Op *, Devicex *);
  std::vector<poplar::Tensor>
  calculateWeightDeltas(poplar::program::Sequence &) const final;
};

class ConvFlipWeightsGradOpx : public Opx {
public:
  ConvFlipWeightsGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVX_HPP_
