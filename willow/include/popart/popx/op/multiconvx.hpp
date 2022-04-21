// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTICONVX_HPP
#define GUARD_NEURALNET_MULTICONVX_HPP

#include <popart/names.hpp>
#include <popart/popx/enigma.hpp>
#include <popart/popx/op/convbasex.hpp>
#include <popart/popx/popopx.hpp>

#include <poplin/MultiConvolution.hpp>

namespace popart {
namespace popx {

class MultiConvOpx : public MultiConvBaseOpx {
public:
  MultiConvOpx(Op *, Devicex *);

  snap::Tensor createWeightsInput(const poplar::DebugNameAndId &dnai,
                                  int convIndex) const final;
  snap::Tensor createDataInput(const poplar::DebugNameAndId &dnai,
                               int convIndex) const final;
  std::vector<snap::Tensor>
  convolve(snap::program::Sequence &,
           const std::vector<snap::Tensor> &) const final;

private:
  std::vector<poplin::multiconv::CreateTensorArgs>
  getCreateTensorArgs(const poplar::DebugNameAndId &dnai) const;
  poplar::OptionFlags getGlobalOptions() const;
};

class MultiConvWeightsGradOpx : public MultiConvWeightsGradBaseOpx {
public:
  MultiConvWeightsGradOpx(Op *, Devicex *);
  std::vector<snap::Tensor>
  calculateWeightDeltas(snap::program::Sequence &) const final;

private:
  poplar::OptionFlags getGlobalOptions() const;
};

} // namespace popx
} // namespace popart

#endif
