// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONVX_HPP
#define GUARD_NEURALNET_CONVX_HPP

#include <popart/names.hpp>
#include <popart/op/conv.hpp>
#include <popart/popx/enigma.hpp>
#include <popart/popx/op/convbasex.hpp>
#include <popart/popx/popopx.hpp>

#include <poplin/Convolution.hpp>

namespace popart {
namespace popx {

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

#endif
