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

  poplar::Tensor createWeightsInput(const poplar::DebugNameAndId &dnai,
                                    int convIndex) const final;
  poplar::Tensor createDataInput(const poplar::DebugNameAndId &dnai,
                                 int convIndex) const final;
  std::vector<poplar::Tensor>
  convolve(poplar::program::Sequence &,
           const std::vector<poplar::Tensor> &) const final;

private:
  std::vector<poplin::multiconv::CreateTensorArgs>
  getCreateTensorArgs(const poplar::DebugNameAndId &dnai) const;
  poplar::OptionFlags getGlobalOptions() const;
};

class MultiConvWeightsGradOpx : public MultiConvWeightsGradBaseOpx {
public:
  MultiConvWeightsGradOpx(Op *, Devicex *);
  std::vector<poplar::Tensor>
  calculateWeightDeltas(poplar::program::Sequence &) const final;

private:
  poplar::OptionFlags getGlobalOptions() const;
};

} // namespace popx
} // namespace popart

#endif
