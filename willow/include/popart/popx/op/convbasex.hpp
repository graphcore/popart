// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONVBASEX_HPP
#define GUARD_NEURALNET_CONVBASEX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

// class MultiConvBaseOp;

namespace popx {

class MultiConvBaseOpx : public Opx {
public:
  MultiConvBaseOpx(Op *op, Devicex *dv) : Opx(op, dv) {}
  poplar::Tensor createInput(InIndex index,
                             const std::string &name) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex index0) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  bool createsEquiv(int, const Opx *, int) const final;
  void grow(poplar::program::Sequence &) const final;

  poplar::OptionFlags getConvOptions(int, std::string pass = "") const;
  std::string getFwdPassFlagString() const;

  virtual std::vector<poplar::Tensor>
  convolve(poplar::program::Sequence &prog,
           const std::vector<poplar::Tensor> &weights) const {
    throw error("No 'convolve' implementation for {}", op_p->opid);
  }
  virtual poplar::Tensor createDataInput(const std::string &name,
                                         int convIndex) const {
    throw error("No 'createDataInput' implementation for {}", op_p->opid);
  }
  virtual poplar::Tensor createWeightsInput(const std::string &name,
                                            int convIndex) const {
    throw error("No 'createWeightsInput' implementation for {}", op_p->opid);
  }
  bool isWeightsInIndex(InIndex) const;
  bool isDataInIndex(InIndex) const;
};

// Returned the canonicalized for of the conv parameters
ConvParameters canonicalizeConvParams(const ConvParameters &param);

// Convert the conv parameters from the fwd conv into the form that
// can be used by the bwd conv
ConvParameters getConvGradParameters(const ConvParameters &fwdParams);

poplin::ConvParams getPoplarConvParams(const ConvParameters &param);
ConvParameters convertPoplarConvParameters(const poplin::ConvParams &popParams);

poplar::Tensor reshapeOnnxWeightsForPoplar(const poplar::Tensor &weights,
                                           std::size_t chansOut,
                                           std::size_t chansIn,
                                           const ConvParameters &params);

class MultiConvWeightsGradBaseOpx : public Opx {
public:
  MultiConvWeightsGradBaseOpx(Op *op, Devicex *dv) : Opx(op, dv) {}
  void grow(poplar::program::Sequence &) const final;
  virtual std::vector<poplar::Tensor>
  calculateWeightDeltas(poplar::program::Sequence &) const {
    throw error("No 'calculateWeightDeltas' implementation for {}", op_p->opid);
  }

  poplar::OptionFlags getConvOptions(int convIndex = 0) const;
};

} // namespace popx
} // namespace popart

#endif
