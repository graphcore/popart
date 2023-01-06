// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_GRUX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_GRUX_HPP_

#include <memory>
#include <set>
#include <poplar/Tensor.hpp>
#include <popnn/Gru.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

#include "popart/popx/debugcontextx.hpp"
#include "popart/vendored/optional.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {

class Op;

namespace popx {
class Devicex;

class GRUOpx : public Opx {
public:
  GRUOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

  static poplar::Tensor reshapePoplibWeightsForOnnx(poplar::Tensor);
  static poplar::Tensor reshapePoplibBiasesForOnnx(poplar::Tensor);

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_GRUX_HPP_
