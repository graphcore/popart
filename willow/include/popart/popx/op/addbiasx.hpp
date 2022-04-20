// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADD_BIASX_HPP
#define GUARD_NEURALNET_ADD_BIASX_HPP

#include "popart/popx/debugcontextx.hpp"
#include <set>
#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/op/reducesumx.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class AddBiasOpx : public PopOpx {
public:
  AddBiasOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;

  std::set<TensorId> mustExistBeforeCreate(int index0) const override;
  InputCreatorType getInputCreatorType(InIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
};

class AddBiasInplaceOpx : public AddBiasOpx {
public:
  AddBiasInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class AddBiasDataGradOpx : public PopOpx {
public:
  AddBiasDataGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class AddBiasBiasGradOpx : public ReduceSumOpx {
public:
  AddBiasBiasGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
