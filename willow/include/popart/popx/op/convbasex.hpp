// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONVBASEX_HPP
#define GUARD_NEURALNET_CONVBASEX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

// class MultiConvBaseOp;

namespace popx {

class MultiConvBaseOpx : public PopOpx {
public:
  MultiConvBaseOpx(Op *op, Devicex *dv) : PopOpx(op, dv) {}
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex index0) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  void grow(poplar::program::Sequence &) const final;

  poplar::OptionFlags getConvOptions(int, std::string pass = "") const;
  std::string getFwdPassFlagString() const;

  virtual std::vector<poplar::Tensor>
  convolve(poplar::program::Sequence &prog,
           const std::vector<poplar::Tensor> &weights) const {
    throw error("No 'convolve' implementation for {}", op_p->opid);
  }
  virtual poplar::Tensor createDataInput(const poplar::DebugNameAndId &dnai,
                                         int convIndex) const {
    throw error("No 'createDataInput' implementation for {}", op_p->opid);
  }
  virtual poplar::Tensor createWeightsInput(const poplar::DebugNameAndId &dnai,
                                            int convIndex) const {
    throw error("No 'createWeightsInput' implementation for {}", op_p->opid);
  }
  bool isWeightsInIndex(InIndex) const;
  bool isDataInIndex(InIndex) const;

  void verifyCacheSizeUnchanged(size_t beforeCacheSize) const;
};

// Returned the canonicalized for of the conv parameters
ConvParameters canonicalizeConvParams(const ConvParameters &param);

// Convert the conv parameters from the fwd conv into the form that
// can be used by the data grad conv
ConvParameters getConvGradParameters(const ConvParameters &fwdParams);

// Convert the conv parameters from the fwd conv into the form that
// can be used by the weights grad conv
ConvParameters getConvWeightUpdateParameters(const ConvParameters &fwdParams);

poplin::ConvParams getPoplarConvParams(const ConvParameters &param);
ConvParameters convertPoplarConvParameters(const poplin::ConvParams &popParams);

poplar::Tensor reshapeOnnxWeightsForPoplar(const poplar::Tensor &weights,
                                           std::size_t chansOut,
                                           std::size_t chansIn,
                                           const ConvParameters &params);

class MultiConvWeightsGradBaseOpx : public PopOpx {
public:
  MultiConvWeightsGradBaseOpx(Op *op, Devicex *dv) : PopOpx(op, dv) {}
  void grow(poplar::program::Sequence &) const final;
  virtual std::vector<poplar::Tensor>
  calculateWeightDeltas(poplar::program::Sequence &) const {
    throw error("No 'calculateWeightDeltas' implementation for {}", op_p->opid);
  }

  poplar::OptionFlags getConvOptions(int convIndex = 0) const;

  void verifyCacheSizeUnchanged(size_t beforeCacheSize) const;
};

} // namespace popx
} // namespace popart

#endif
