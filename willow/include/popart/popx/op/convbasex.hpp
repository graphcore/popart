// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVBASEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVBASEX_HPP_

#include <cstddef>
#include <set>
#include <string>
#include <vector>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <poplin/ConvParams.hpp>
#include <popart/op/convbase.hpp>
#include <popart/popx/opx.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/popx/debugcontextx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {

// class MultiConvBaseOp;

namespace popx {
class Devicex;

class MultiConvBaseOpx : public Opx {
public:
  MultiConvBaseOpx(Op *op, Devicex *dv) : Opx(op, dv) {}
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex index0) const final;
  InputCreatorType getInputCreatorType(InIndex) const override;
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

class MultiConvWeightsGradBaseOpx : public Opx {
public:
  MultiConvWeightsGradBaseOpx(Op *op, Devicex *dv) : Opx(op, dv) {}
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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONVBASEX_HPP_
