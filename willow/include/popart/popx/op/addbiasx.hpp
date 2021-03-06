// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADD_BIASX_HPP
#define GUARD_NEURALNET_ADD_BIASX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/identityx.hpp>
#include <popart/popx/op/reducesumx.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class AddBiasOpx : public PopOpx {
public:
  AddBiasOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

  std::set<TensorId> mustExistBeforeCreate(int index0) const override;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
};

class AddBiasInplaceOpx : public AddBiasOpx {
public:
  AddBiasInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class AddBiasDataGradOpx : public PopOpx {
public:
  AddBiasDataGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class AddBiasBiasGradOpx : public ReduceSumOpx {
public:
  AddBiasBiasGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
