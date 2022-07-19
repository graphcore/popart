// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_HOSTCOPYX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_HOSTCOPYX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <snap/Tensor.hpp>
#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/popopx.hpp>

#include "popart/names.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

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
  void grow(snap::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class HostLoadInplaceOpx : public HostLoadOpx {
public:
  HostLoadInplaceOpx(Op *, Devicex *);
};

class HostStoreOpx : public HostBaseOpx {
public:
  HostStoreOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_HOSTCOPYX_HPP_
