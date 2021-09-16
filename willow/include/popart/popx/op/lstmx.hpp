// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTMX_HPP
#define GUARD_NEURALNET_LSTMX_HPP

#include <snap/Tensor.hpp>
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
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

  static snap::Tensor reshapePoplibWeightsForOnnx(snap::Tensor, bool transpose);
  static popnn::lstm::LstmParams createLSTMParams(const LSTMOp &,
                                                  const snap::Tensor &);

private:
  void growBias(snap::program::Sequence &) const;
  popnn::lstm::LstmWeights getLSTMWeights() const;
  popnn::lstm::LstmState getInitialState() const;
  snap::Tensor createLSTMInput() const;
  void prepareInitialState(popnn::lstm::LstmState &,
                           snap::program::Sequence &) const;
  void prepareWeights(snap::program::Sequence &) const;
  snap::Tensor getInput(snap::program::Sequence &) const;
  std::unique_ptr<snap::Tensor> createIntermediate() const;
  void reshapeAndInsert(OutIndex index, const snap::Tensor &) const;
  bool inputCreated(InIndex) const;
  snap::Tensor getSeqLens() const;

  mutable nonstd::optional<popnn::lstm::LstmWeights> weights;
  mutable nonstd::optional<popnn::lstm::LstmState> initial_state;
  mutable std::set<InIndex> createdInputs;
};

class LSTMGradOpx : public PopOpx {
public:
  LSTMGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  snap::Tensor getCellStateGrad() const;
  snap::Tensor getHiddenStateGrad() const;
  snap::Tensor getSeqLens() const;
};

} // namespace popx
} // namespace popart

#endif
