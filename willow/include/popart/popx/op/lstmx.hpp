// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTMX_HPP
#define GUARD_NEURALNET_LSTMX_HPP

#include <poplar/Tensor.hpp>
#include <popnn/Lstm.hpp>
#include <popart/names.hpp>
#include <popart/op/lstm.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class LSTMOp;

namespace popx {

class LSTMOpx : public PopOpx {
public:
  LSTMOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

  static poplar::Tensor reshapePoplibWeightsForOnnx(poplar::Tensor,
                                                    bool transpose);
  static popnn::lstm::LstmParams createLSTMParams(const LSTMOp &,
                                                  const poplar::Tensor &);

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

class LSTMGradOpx : public PopOpx {
public:
  LSTMGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor getCellStateGrad() const;
  poplar::Tensor getHiddenStateGrad() const;
  poplar::Tensor getSeqLens() const;
};

} // namespace popx
} // namespace popart

#endif
