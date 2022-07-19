// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_REMOTEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_REMOTEX_HPP_

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

class RemoteBaseOpx : public ExchangeBaseOpx {
public:
  RemoteBaseOpx(Op *, Devicex *);
};

class RemoteStoreOpx : public RemoteBaseOpx {
public:
  RemoteStoreOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class RemoteLoadOpx : public RemoteBaseOpx {
public:
  RemoteLoadOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class RemoteLoadInplaceOpx : public RemoteLoadOpx {
public:
  RemoteLoadInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_REMOTEX_HPP_
