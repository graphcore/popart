// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADDBIASX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADDBIASX_HPP_

#include <set>
#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/op/reducesumx.hpp>
#include <popart/popx/opx.hpp>

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

class AddBiasOpx : public Opx {
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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADDBIASX_HPP_
