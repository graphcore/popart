// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRUX_HPP
#define GUARD_NEURALNET_GRUX_HPP

#include <popnn/Gru.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class GRUOp;

namespace popx {

class GRUOpx : public Opx {
public:
  GRUOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

  static popnn::gru::GruParams createGRUParams(const GRUOp &);
  static poplar::Tensor reshapePoplibWeightsForOnnx(poplar::Tensor,
                                                    bool transpose);

private:
  void growBias(poplar::program::Sequence &) const;
  popnn::gru::GruParams createGRUParams() const;
  popnn::gru::GruWeights getGRUWeights() const;
  poplar::Tensor getInitialState() const;
  poplar::Tensor createGRUInput() const;
  void prepareWeights(poplar::program::Sequence &) const;
  void prepareInitialState(poplar::Tensor &init_state_h,
                           poplar::program::Sequence &prog) const;
  poplar::Tensor getInput(poplar::program::Sequence &) const;
  std::unique_ptr<poplar::Tensor> createIntermediate() const;
  void reshapeAndInsert(OutIndex index, const poplar::Tensor &) const;
  bool inputCreated(InIndex) const;

  // These are mutable due to the way that popnn creates the input weights
  mutable nonstd::optional<popnn::gru::GruWeights> weights;
  mutable nonstd::optional<poplar::Tensor> initial_state_h;
  mutable std::set<InIndex> createdInputs;
};

class GRUGradOpx : public Opx {
public:
  GRUGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor getHiddenStateGrad() const;

  popnn::gru::GruParams createGRUParams() const;
};

} // namespace popx
} // namespace popart

#endif
