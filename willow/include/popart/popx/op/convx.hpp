// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONVX_HPP
#define GUARD_NEURALNET_CONVX_HPP

#include <popart/names.hpp>
#include <popart/popx/enigma.hpp>
#include <popart/popx/opx.hpp>

#include <poplin/Convolution.hpp>

namespace popart {

class ConvOp;
class ConvWeightsGradOp;
class ConvDataGradOp;

namespace popx {

class ConvOpx : public Opx {
public:
  ConvOpx(Op *, Devicex *);
  poplar::Tensor createInput(InIndex index,
                             const std::string &name) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  bool createsEquiv(int, const Opx *, int) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex index0) const final;
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::OptionFlags getOptions() const;
};

class ConvWeightsGradOpx : public Opx {
public:
  ConvWeightsGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ConvFlipWeightsGradOpx : public Opx {
public:
  ConvFlipWeightsGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
};

// Helper functions to convert the ConvParameter

// Returned the canonicalized for of the conv parameters
ConvParameters canonicalizeConvParams(const ConvParameters &param);

// Convert the conv parameters from the fwd conv into the form that
// can be used by the bwd conv
ConvParameters getConvGradParameters(const ConvParameters &fwdParams);

} // namespace popx
} // namespace popart

#endif
