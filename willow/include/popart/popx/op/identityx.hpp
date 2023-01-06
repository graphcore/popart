// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_IDENTITYX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_IDENTITYX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class IdentityOpx : public ElementWiseUnaryOpx {
public:
  IdentityOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityInplaceOpx : public Opx {
public:
  IdentityInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityGradOpx : public ElementWiseUnaryOpx {
public:
  IdentityGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityLossOpx : public Opx {
public:
  IdentityLossOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;

  poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

class IdentityLossGradOpx : public Opx {
public:
  IdentityLossGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  bool outputCreatedExternally(OutIndex) const final { return true; }
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_IDENTITYX_HPP_
