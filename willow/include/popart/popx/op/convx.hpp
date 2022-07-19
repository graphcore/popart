// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <snap/Tensor.hpp>
#include <vector>
#include <popart/popx/op/convbasex.hpp>
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

class ConvOpx : public MultiConvBaseOpx {
public:
  ConvOpx(Op *, Devicex *);
  snap::Tensor createWeightsInput(const poplar::DebugNameAndId &dnai,
                                  int convIndex) const final;
  snap::Tensor createDataInput(const poplar::DebugNameAndId &dnai,
                               int convIndex) const final;
  std::vector<snap::Tensor>
  convolve(snap::program::Sequence &,
           const std::vector<snap::Tensor> &weights) const final;
};

class ConvWeightsGradOpx : public MultiConvWeightsGradBaseOpx {
public:
  ConvWeightsGradOpx(Op *, Devicex *);
  std::vector<snap::Tensor>
  calculateWeightDeltas(snap::program::Sequence &) const final;
};

class ConvFlipWeightsGradOpx : public PopOpx {
public:
  ConvFlipWeightsGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVX_HPP_
