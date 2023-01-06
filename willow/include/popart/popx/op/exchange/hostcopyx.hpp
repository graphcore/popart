// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_HOSTCOPYX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_HOSTCOPYX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/opx.hpp>

#include "popart/names.hpp"
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

class HostBaseOpx : public ExchangeBaseOpx {
public:
  HostBaseOpx(Op *, Devicex *);
};

class HostLoadOpx : public HostBaseOpx {
public:
  HostLoadOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class HostLoadInplaceOpx : public HostLoadOpx {
public:
  HostLoadInplaceOpx(Op *, Devicex *);
};

class HostStoreOpx : public HostBaseOpx {
public:
  HostStoreOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_HOSTCOPYX_HPP_
