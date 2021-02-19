// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADD_BIASX_HPP
#define GUARD_NEURALNET_ADD_BIASX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/identityx.hpp>
#include <popart/popx/op/reducesumx.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class AddBiasOpx : public Opx {
public:
  AddBiasOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

  std::vector<TensorId> mustExistBeforeCreate(int index0) const override;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  bool createsEquiv(int index0, const Opx *opx1, int index1) const final;
};

class AddBiasInplaceOpx : public AddBiasOpx {
public:
  AddBiasInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class AddBiasDataGradOpx : public Opx {
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
