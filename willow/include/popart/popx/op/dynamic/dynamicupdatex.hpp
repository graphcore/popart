// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICUPDATEX_HPP
#define GUARD_NEURALNET_DYNAMICUPDATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class DynamicUpdateOpx : public PopOpx {
public:
  DynamicUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const override;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
  virtual poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                                       const poplar::Tensor &) const;
};

class DynamicUpdateInplaceOpx : public DynamicUpdateOpx {
public:
  DynamicUpdateInplaceOpx(Op *, Devicex *);
  poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                               const poplar::Tensor &) const override;
};

} // namespace popx
} // namespace popart

#endif
