// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LSTMX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LSTMX_HPP_

#include <memory>
#include <set>
#include <poplar/Tensor.hpp>
#include <popnn/Lstm.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>
#include <popart/vendored/optional.hpp>

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

class LSTMOpx : public Opx {
public:
  LSTMOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

  static poplar::Tensor reshapePoplibWeightsForOnnx(poplar::Tensor,
                                                    bool transpose);
  popnn::lstm::LstmParams createLSTMParams() const;

private:
  void growBias(poplar::program::Sequence &) const;
  popnn::lstm::LstmWeights getLSTMWeights() const;
  popnn::lstm::LstmState getInitialState() const;
  poplar::Tensor createLSTMInput() const;
  void prepareInitialState(popnn::lstm::LstmState &,
                           poplar::program::Sequence &) const;
  void prepareWeights(poplar::program::Sequence &) const;
  poplar::Tensor getInput(poplar::program::Sequence &) const;
  std::unique_ptr<poplar::Tensor> createIntermediate() const;
  void reshapeAndInsert(OutIndex index, const poplar::Tensor &) const;
  bool inputCreated(InIndex) const;
  poplar::Tensor getSeqLens() const;

  mutable nonstd::optional<popnn::lstm::LstmWeights> weights;
  mutable nonstd::optional<popnn::lstm::LstmState> initial_state;
  mutable std::set<InIndex> createdInputs;
};

class LSTMGradOpx : public Opx {
public:
  LSTMGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor getCellStateGrad() const;
  poplar::Tensor getHiddenStateGrad() const;
  poplar::Tensor getSeqLens() const;
  popnn::lstm::LstmParams createLSTMParams() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LSTMX_HPP_
