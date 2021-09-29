// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRUX_HPP
#define GUARD_NEURALNET_GRUX_HPP

#include <popnn/Gru.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

class GRUOp;

namespace popx {

class GRUOpx : public PopOpx {
public:
  GRUOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

  static popnn::gru::GruParams createGRUParams(const GRUOp &);
  static snap::Tensor reshapePoplibWeightsForOnnx(snap::Tensor);
  static snap::Tensor reshapePoplibBiasesForOnnx(snap::Tensor);

private:
  void growBias(snap::program::Sequence &) const;
  popnn::gru::GruParams createGRUParams() const;
  popnn::gru::GruWeights getGRUWeights() const;
  snap::Tensor getInitialState() const;
  snap::Tensor createGRUInput() const;
  void prepareWeights(snap::program::Sequence &) const;
  void prepareInitialState(snap::Tensor &init_state_h,
                           snap::program::Sequence &prog) const;
  snap::Tensor getInput(snap::program::Sequence &) const;
  std::unique_ptr<snap::Tensor> createIntermediate() const;
  void reshapeAndInsert(OutIndex index, const snap::Tensor &) const;
  bool inputCreated(InIndex) const;

  // These are mutable due to the way that popnn creates the input weights
  mutable nonstd::optional<popnn::gru::GruWeights> weights;
  mutable nonstd::optional<snap::Tensor> initial_state_h;
  mutable std::set<InIndex> createdInputs;
};

class GRUGradOpx : public PopOpx {
public:
  GRUGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  snap::Tensor getHiddenStateGrad() const;

  popnn::gru::GruParams createGRUParams() const;
};

} // namespace popx
} // namespace popart

#endif
