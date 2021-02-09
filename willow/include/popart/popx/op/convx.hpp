// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONVX_HPP
#define GUARD_NEURALNET_CONVX_HPP

#include <popart/names.hpp>
#include <popart/op/conv.hpp>
#include <popart/popx/enigma.hpp>
#include <popart/popx/op/convbasex.hpp>
#include <popart/popx/opx.hpp>

#include <poplin/Convolution.hpp>

namespace popart {
namespace popx {

class ConvOpx : public MultiConvBaseOpx {
public:
  ConvOpx(Op *, Devicex *);
  poplar::Tensor createWeightsInput(const poplar::DebugNameAndId &dnai,
                                    int convIndex) const final;
  poplar::Tensor createDataInput(const poplar::DebugNameAndId &dnai,
                                 int convIndex) const final;
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

#endif
